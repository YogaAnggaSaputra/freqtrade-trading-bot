"""
attribution.py
==============
Feedback loop Fase 2 — Attribution Engine.

Konsumsi event TRADE_CLOSED (Redis pub/sub) + reconciliation sweep dari DB
(antisipasi Redis at-most-once). Untuk tiap trade:
  - signal_correct: apakah arah/RR prediksi sesuai outcome aktual
  - drift_contribution: |actual_rr - predicted_rr|
  - rolling drift metric: MAE(predicted_rr, actual_rr) window N terakhir
  - IF drift_threshold_breached OR batch_size_reached → publish RETRAIN_TRIGGER

Attribution ditulis ke attribution_results, dan trade_outcomes ditandai
processed_by_attribution=true (idempotent).
"""
from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from typing import Any, Dict, Optional

from sqlalchemy import select, update, func
from sqlalchemy import text

from shared.db.models import TradeOutcome, AttributionResult
from shared.db.session import AsyncSessionLocal

logger = logging.getLogger("loss_analyzer.attribution")

# Threshold konfigurabel via env (dilarang hardcode nilai bisnis)
DRIFT_MAE_THRESHOLD = float(os.getenv("FB_DRIFT_MAE_THRESHOLD", "0.5"))
RETRAIN_BATCH_SIZE = int(os.getenv("FB_RETRAIN_BATCH_SIZE", "50"))
DRIFT_WINDOW = int(os.getenv("FB_DRIFT_WINDOW", "30"))


class AttributionEngine:
    """Attribution + drift + retrain trigger."""

    def __init__(self, message_bus=None):
        self.message_bus = message_bus

    async def process_trade(self, outcome: Dict[str, Any]) -> Optional[str]:
        """Proses satu trade outcome → tulis attribution_results.

        Idempotent: kalau trade_id sudah processed, skip. Return
        attribution verdict string atau None kalau di-skip.
        """
        trade_id = int(outcome["trade_id"])
        async with AsyncSessionLocal() as db:
            # Skip kalau sudah diproses (idempotent — penting untuk sweep+event)
            existing = await db.execute(
                select(AttributionResult.id).where(AttributionResult.trade_id == trade_id)
            )
            if existing.scalar_one_or_none() is not None:
                return None

            predicted_rr = outcome.get("predicted_rr")
            actual_rr = outcome.get("actual_rr")
            pnl_pct = float(outcome.get("pnl_pct") or 0.0)
            regime = outcome.get("regime_at_entry")
            entry = outcome.get("entry_conditions") or {}
            ml_signal = entry.get("ml_signal")
            side = entry.get("side")

            # signal_correct:
            #  - jika ada predicted_rr: benar bila actual_rr >= predicted_rr (target tercapai)
            #  - fallback: benar bila arah ML sesuai profit (BUY→profit long, SELL→profit short)
            signal_correct = self._judge_signal(
                predicted_rr, actual_rr, pnl_pct, ml_signal, side
            )

            # drift_contribution: absolut deviasi RR (kalau tersedia)
            drift_contribution = None
            if predicted_rr is not None and actual_rr is not None:
                drift_contribution = abs(float(actual_rr) - float(predicted_rr))

            attribution = AttributionResult(
                trade_id=trade_id,
                signal_correct=signal_correct,
                drift_contribution=drift_contribution,
                regime=regime,
                feature_importance_snapshot=None,  # diisi Fase 3 (retrainer SHAP)
                ml_recommendation=ml_signal,
                pnl_pct=pnl_pct,
            )
            db.add(attribution)

            # Tandai processed (reconciliation flag)
            await db.execute(
                update(TradeOutcome)
                .where(TradeOutcome.trade_id == trade_id)
                .values(processed_by_attribution=True)
            )
            await db.commit()

        verdict = "correct" if signal_correct else "incorrect"
        logger.info(
            "attribution trade_id=%s regime=%s signal=%s drift=%s",
            trade_id, regime, verdict, drift_contribution
        )

        # Evaluasi apakah perlu retrain
        await self._maybe_trigger_retrain()
        return verdict

    @staticmethod
    def _judge_signal(predicted_rr, actual_rr, pnl_pct, ml_signal, side) -> bool:
        """Apakah sinyal 'benar'? Prioritas: RR target, fallback arah."""
        if predicted_rr is not None and actual_rr is not None:
            # Sinyal dianggap benar kalau realisasi RR minimal setengah target
            # (bukan harus penuh — TP partial pun sudah "arah benar").
            try:
                return float(actual_rr) >= 0.5 * float(predicted_rr)
            except (TypeError, ValueError):
                pass
        # Fallback: profit positif = arah benar
        return pnl_pct >= 0

    async def _maybe_trigger_retrain(self) -> None:
        """Cek drift MAE window + batch size unprocessed→trigger retrain."""
        async with AsyncSessionLocal() as db:
            # Rolling MAE(predicted_rr, actual_rr) atas N attribution terakhir
            mae_row = await db.execute(text("""
                SELECT AVG(ABS(drift_contribution)) AS mae, COUNT(*) AS n
                FROM (
                    SELECT drift_contribution FROM attribution_results
                    WHERE drift_contribution IS NOT NULL
                    ORDER BY created_at DESC LIMIT :win
                ) t
            """), {"win": DRIFT_WINDOW})
            row = mae_row.first()
            drift_mae = float(row.mae) if row and row.mae is not None else 0.0
            n_window = int(row.n) if row and row.n is not None else 0

            # Hitung berapa trade baru sejak retrain terakhir (audit trail)
            since_row = await db.execute(text("""
                SELECT COUNT(*) FROM attribution_results a
                WHERE a.created_at > COALESCE(
                    (SELECT MAX(triggered_at) FROM retrain_jobs), '1970-01-01'::timestamptz
                )
            """))
            new_trades = int(since_row.scalar() or 0)

        drift_breached = n_window >= DRIFT_WINDOW and drift_mae > DRIFT_MAE_THRESHOLD
        batch_reached = new_trades >= RETRAIN_BATCH_SIZE

        if not (drift_breached or batch_reached):
            return

        reason = "drift" if drift_breached else "batch_size"
        logger.warning(
            "RETRAIN_TRIGGER: reason=%s drift_mae=%.3f new_trades=%d",
            reason, drift_mae, new_trades
        )
        if self.message_bus is not None:
            try:
                from shared.messaging import Channels
                await self.message_bus.publish(Channels.RETRAIN_TRIGGER, {
                    "trigger_reason": reason,
                    "drift_mae": round(drift_mae, 4),
                    "new_trades": new_trades,
                    "triggered_at": datetime.now(UTC).isoformat(),
                })
            except Exception as exc:  # noqa: BLE001
                logger.warning("publish RETRAIN_TRIGGER failed: %s", exc)

    async def reconciliation_sweep(self) -> int:
        """Proses trade_outcomes yang belum ter-attribution (Redis miss).

        Return jumlah trade yang diproses ulang.
        """
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(TradeOutcome).where(
                    TradeOutcome.processed_by_attribution.is_(False)
                ).limit(500)
            )
            unprocessed = result.scalars().all()

        count = 0
        for to in unprocessed:
            outcome = {
                "trade_id": to.trade_id,
                "pair": to.pair,
                "predicted_rr": float(to.predicted_rr) if to.predicted_rr is not None else None,
                "actual_rr": float(to.actual_rr) if to.actual_rr is not None else None,
                "pnl_pct": float(to.pnl_pct),
                "regime_at_entry": to.regime_at_entry,
                "entry_conditions": to.entry_conditions or {},
            }
            try:
                await self.process_trade(outcome)
                count += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("sweep process trade_id=%s failed: %s", to.trade_id, exc)
        if count:
            logger.info("reconciliation_sweep processed %d trades", count)
        return count
