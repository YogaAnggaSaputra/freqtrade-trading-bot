"""
analyzer.py
============
Loss Analyzer — mendeteksi pola loss sistematis, drift pasar,
dan kegagalan teknis dari trade dossier.

Mengklasifikasikan setiap trade loss dan menulis incident ke database.
"""
from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import and_, select

from shared.db.models import Incident, MarketCandle, TradeDossier
from shared.db.session import AsyncSessionLocal

logger = logging.getLogger("loss_analyzer.analyzer")

# Klasifikasi loss berdasarkan data dossier
LOSS_CLASSES = {
    "regime_mismatch": "Strategi tidak cocok untuk regime pasar saat ini",
    "poor_timing": "Entry/exit timing buruk",
    "sl_noise": "Stop-loss terkena noise market normal",
    "feature_drift": "Fitur bergeser jauh dari distribusi training",
    "execution_slippage": "Slippage eksekusi signifikan",
    "technical_failure": "Kegagalan teknis (timeout, mismatch)",
    "unknown": "Penyebab loss tidak teridentifikasi",
}


class LossAnalyzer:
    """Menganalisis dossier trade dan mengklasifikasikan kerugian."""

    async def classify_trade_loss(
        self, dossier: TradeDossier
    ) -> str:
        """Tentukan klasifikasi loss untuk satu trade."""
        if float(dossier.realized_pnl) >= 0:
            return "winner"

        # Cek insiden teknis
        incidents = dossier.technical_incidents or []
        if any(
            i.get("type") in ("api_timeout", "order_mismatch", "reconnect")
            for i in incidents
        ):
            return "technical_failure"

        # Cek exit reason
        exit_reason = (dossier.exit_reason or "").lower()
        if "stoploss" in exit_reason or "stop_loss" in exit_reason:
            # Evaluasi apakah SL kena noise
            sl_tp = dossier.sl_tp or {}
            entry = dossier.entry or {}
            entry_price = float(entry.get("price", 0))
            sl_price = float(sl_tp.get("stop_loss", 0))
            loss_pct = abs(entry_price - sl_price) / entry_price if entry_price > 0 else 0
            if loss_pct < 0.008:  # SL < 0.8% dari entry
                return "sl_noise"
            return "regime_mismatch"

        # Cek regime dari feature snapshot
        feature_snap = dossier.feature_snapshot or {}
        regime = feature_snap.get("regime", "unknown")
        signal_meta = dossier.entry_signal or {}
        signal_regime = signal_meta.get("regime", "unknown")
        if regime != signal_regime and regime != "unknown":
            return "regime_mismatch"

        # Cek timing (exit terlalu cepat setelah entry)
        # Normalisasi tz-aware → naive UTC agar subtraction tidak crash.
        if dossier.created_at and dossier.closed_at:
            created = dossier.created_at.replace(tzinfo=None) if dossier.created_at.tzinfo else dossier.created_at
            closed = dossier.closed_at.replace(tzinfo=None) if dossier.closed_at.tzinfo else dossier.closed_at
            hold_time = closed - created
            if hold_time.total_seconds() < 300:  # < 5 menit
                return "poor_timing"

        return "unknown"

    async def run_batch_classification(
        self,
        days: int = 7,
        strategy_version: str | None = None,
    ) -> dict[str, Any]:
        """
        Klasifikasikan semua trade loss yang belum terklasifikasi.
        Update kolom loss_classification di database.
        """
        since = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=days)
        classified_count = 0
        summary: dict[str, int] = {}

        async with AsyncSessionLocal() as db:
            stmt = select(TradeDossier).where(
                and_(
                    TradeDossier.created_at >= since,
                    TradeDossier.realized_pnl < 0,
                    TradeDossier.loss_classification.is_(None),
                )
            )
            if strategy_version:
                stmt = stmt.where(TradeDossier.strategy_version == strategy_version)

            result = await db.execute(stmt)
            dossiers = result.scalars().all()

            for dossier in dossiers:
                classification = await self.classify_trade_loss(dossier)
                dossier.loss_classification = classification
                summary[classification] = summary.get(classification, 0) + 1
                classified_count += 1

            await db.commit()

        logger.info(
            "Classified %d loss trades: %s", classified_count, summary
        )
        return {
            "classified_count": classified_count,
            "summary": summary,
        }

    async def detect_systematic_patterns(
        self,
        days: int = 14,
        strategy_version: str | None = None,
        min_samples: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Deteksi pola loss sistematis yang perlu dilaporkan sebagai incident.
        """
        since = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=days)
        patterns = []

        async with AsyncSessionLocal() as db:
            # Ambil semua trade (bukan cuma loss) agar streak bisa di-reset oleh winner.
            # Kalau cuma loss data, streak = len(losses) → selalu false-positive.
            stmt = select(TradeDossier).where(
                TradeDossier.created_at >= since,
            )
            if strategy_version:
                stmt = stmt.where(TradeDossier.strategy_version == strategy_version)
            result = await db.execute(stmt.order_by(TradeDossier.created_at))
            all_trades = result.scalars().all()

        if len(all_trades) < min_samples:
            return []

        losses = [d for d in all_trades if float(d.realized_pnl) < 0]

        # Pattern 1: Regime mismatch dominan
        regime_mismatch_count = sum(
            1 for d in losses if d.loss_classification == "regime_mismatch"
        )
        if regime_mismatch_count / len(losses) > 0.5:
            patterns.append({
                "pattern": "dominant_regime_mismatch",
                "severity": "high",
                "description": f"{regime_mismatch_count}/{len(losses)} losses classified as regime_mismatch",
                "recommendation": "Review ADX filter threshold and regime detection logic",
            })

        # Pattern 2: SL noise dominan
        sl_noise_count = sum(
            1 for d in losses if d.loss_classification == "sl_noise"
        )
        if sl_noise_count / len(losses) > 0.4:
            patterns.append({
                "pattern": "sl_noise_dominant",
                "severity": "medium",
                "description": f"{sl_noise_count}/{len(losses)} losses due to SL noise (SL too tight)",
                "recommendation": "Increase ATR multiplier for stop-loss placement",
            })

        # Pattern 3: Loss streak — pakai all_trades (bukan loss-only) agar winner reset streak
        max_streak = 0
        current_streak = 0
        for d in all_trades:
            if float(d.realized_pnl) < 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        if max_streak >= 6:
            patterns.append({
                "pattern": "extended_loss_streak",
                "severity": "high",
                "description": f"Loss streak of {max_streak} detected",
                "recommendation": "Consider pausing trading and running diagnostic",
            })

        return patterns

    async def create_incident_if_needed(
        self,
        pattern: dict[str, Any],
        strategy_version: str,
    ) -> str | None:
        """Buat incident record bila pattern terdeteksi cukup serius."""
        if pattern.get("severity") not in ("high", "critical"):
            return None

        incident_id = f"INC-{uuid.uuid4().hex[:8].upper()}"
        async with AsyncSessionLocal() as db:
            incident = Incident(
                incident_id=incident_id,
                incident_type=pattern["pattern"],
                severity=pattern["severity"],
                title=f"[{strategy_version}] {pattern['pattern'].replace('_', ' ').title()}",
                description=pattern.get("description", ""),
                related_ids={"strategy_version": strategy_version},
                status="open",
                created_at=datetime.now(UTC),
            )
            db.add(incident)
            await db.commit()

        logger.info("Created incident: %s (%s)", incident_id, pattern["pattern"])
        return incident_id

    async def compute_drift_score(
        self,
        pair: str,
        days: int = 7,
    ) -> dict[str, Any]:
        """
        Hitung drift score sederhana berdasarkan perubahan distribusi
        fitur market (ATR, ADX, volatilitas) vs baseline.
        """
        since = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=days)
        baseline_since = since - timedelta(days=days * 3)

        async with AsyncSessionLocal() as db:
            # Recent candles
            recent_result = await db.execute(
                select(MarketCandle).where(
                    and_(
                        MarketCandle.pair == pair,
                        MarketCandle.timestamp >= since,
                    )
                ).order_by(MarketCandle.timestamp)
            )
            recent_candles = recent_result.scalars().all()

            # Baseline candles
            baseline_result = await db.execute(
                select(MarketCandle).where(
                    and_(
                        MarketCandle.pair == pair,
                        MarketCandle.timestamp >= baseline_since,
                        MarketCandle.timestamp < since,
                    )
                ).order_by(MarketCandle.timestamp)
            )
            baseline_candles = baseline_result.scalars().all()

        if not recent_candles or not baseline_candles:
            return {"drift_score": 0.0, "details": "Insufficient data"}

        # Hitung volatilitas sederhana (std dari log return)
        import math

        def log_returns(candles):
            closes = [float(c.close) for c in candles]
            if len(closes) < 2:
                return []
            return [
                math.log(closes[i] / closes[i - 1])
                for i in range(1, len(closes))
                if closes[i - 1] > 0
            ]

        recent_lr = log_returns(recent_candles)
        baseline_lr = log_returns(baseline_candles)

        if not recent_lr or not baseline_lr:
            return {"drift_score": 0.0, "details": "Insufficient returns data"}

        from shared.quant.stats_advanced import kolmogorov_smirnov_2sample, wasserstein_distance_1d
        import statistics

        recent_vol = statistics.stdev(recent_lr) if len(recent_lr) > 1 else 0
        baseline_vol = statistics.stdev(baseline_lr) if len(baseline_lr) > 1 else 0

        # Kolmogorov-Smirnov & Wasserstein drift metrics
        ks_result = kolmogorov_smirnov_2sample(recent_lr, baseline_lr)
        w_dist = wasserstein_distance_1d(recent_lr, baseline_lr)
        drift_score = ks_result["d_statistic"]

        return {
            "pair": pair,
            "drift_score": round(drift_score, 4),
            "wasserstein_distance": round(w_dist, 6),
            "ks_p_value": ks_result["p_value_approx"],
            "recent_volatility": round(recent_vol, 6),
            "baseline_volatility": round(baseline_vol, 6),
            "recent_candle_count": len(recent_candles),
            "baseline_candle_count": len(baseline_candles),
            "is_drifted": ks_result["is_drifted"],
        }
