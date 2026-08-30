"""
evidence_collector.py
======================
Mengumpulkan data dari database untuk dianalisis oleh Hermes Agent.
Semua operasi bersifat read-only — Hermes tidak pernah menulis ke DB secara langsung.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select

from shared.db.models import (
    Deployment,
    Incident,
    Prediction,
    Proposal,
    TradeDossier,
)
from shared.db.session import AsyncSessionLocal

logger = logging.getLogger("hermes.evidence_collector")


class EvidenceCollector:
    """Read-only aggregator of trading evidence for Hermes Agent."""

    async def get_recent_trade_dossiers(
        self,
        days: int = 30,
        strategy_version: str | None = None,
        min_trades: int = 10,
    ) -> list[dict[str, Any]]:
        """Ambil dossier trade terbaru dari database."""
        since = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=days)
        async with AsyncSessionLocal() as db:
            stmt = select(TradeDossier).where(TradeDossier.created_at >= since)
            if strategy_version:
                stmt = stmt.where(TradeDossier.strategy_version == strategy_version)
            stmt = stmt.order_by(TradeDossier.created_at.desc())
            result = await db.execute(stmt)
            dossiers = result.scalars().all()

        if len(dossiers) < min_trades:
            logger.info(
                "Insufficient dossiers: %d < %d required", len(dossiers), min_trades
            )

        return [_dossier_to_dict(d) for d in dossiers]

    async def get_loss_summary(
        self,
        days: int = 14,
        strategy_version: str | None = None,
    ) -> dict[str, Any]:
        """Hitung statistik loss agregat untuk periode tertentu."""
        since = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=days)
        async with AsyncSessionLocal() as db:
            stmt = select(TradeDossier).where(TradeDossier.created_at >= since)
            if strategy_version:
                stmt = stmt.where(TradeDossier.strategy_version == strategy_version)
            result = await db.execute(stmt)
            dossiers = result.scalars().all()

        if not dossiers:
            return {
                "sample_size": 0,
                "net_pnl": 0,
                "win_rate": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "max_drawdown": 0,
                "loss_streak": 0,
                "regime_breakdown": {},
                "exit_reason_breakdown": {},
            }

        pnls = [float(d.realized_pnl) for d in dossiers]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]

        # Hitung max drawdown
        equity = 0.0
        peak = 0.0
        max_dd = 0.0
        for p in sorted(pnls):
            equity += p
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)

        # Hitung loss streak
        max_streak = 0
        current_streak = 0
        for p in pnls:
            if p <= 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        # Breakdown per regime
        regime_stats: dict[str, dict] = {}
        for d in dossiers:
            regime = d.market_regime or "unknown"
            if regime not in regime_stats:
                regime_stats[regime] = {"count": 0, "net_pnl": 0.0}
            regime_stats[regime]["count"] += 1
            regime_stats[regime]["net_pnl"] += float(d.realized_pnl)

        # Breakdown per exit_reason
        exit_stats: dict[str, int] = {}
        for d in dossiers:
            reason = d.exit_reason or "unknown"
            exit_stats[reason] = exit_stats.get(reason, 0) + 1

        return {
            "sample_size": len(dossiers),
            "net_pnl": round(sum(pnls), 6),
            "win_rate": round(len(winners) / len(pnls), 4) if pnls else 0,
            "avg_win": round(sum(winners) / len(winners), 6) if winners else 0,
            "avg_loss": round(sum(losers) / len(losers), 6) if losers else 0,
            "max_drawdown": round(max_dd, 4),
            "loss_streak": max_streak,
            "regime_breakdown": regime_stats,
            "exit_reason_breakdown": exit_stats,
        }

    async def get_regime_performance(
        self, days: int = 30
    ) -> dict[str, dict[str, Any]]:
        """Hitung performa per market regime."""
        dossiers = await self.get_recent_trade_dossiers(days=days)
        regime_map: dict[str, list[float]] = {}
        for d in dossiers:
            regime = d.get("market_regime", "unknown")
            regime_map.setdefault(regime, []).append(d.get("realized_pnl", 0.0))

        result = {}
        for regime, pnls in regime_map.items():
            winners = [p for p in pnls if p > 0]
            result[regime] = {
                "sample_size": len(pnls),
                "net_pnl": round(sum(pnls), 6),
                "win_rate": round(len(winners) / len(pnls), 4) if pnls else 0,
                "avg_pnl": round(sum(pnls) / len(pnls), 6) if pnls else 0,
            }
        return result

    async def get_active_strategy_version(self) -> str | None:
        """Dapatkan versi strategi yang saat ini aktif."""
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(Deployment)
                .where(Deployment.status == "active")
                .order_by(Deployment.deployed_at.desc())
                .limit(1)
            )
            deployment = result.scalar_one_or_none()
        return deployment.strategy_version if deployment else None

    async def get_recent_incidents(self, days: int = 7) -> list[dict[str, Any]]:
        """Ambil insiden teknis terbaru."""
        since = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=days)
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(Incident)
                .where(Incident.created_at >= since)
                .order_by(Incident.created_at.desc())
            )
            incidents = result.scalars().all()
        return [
            {
                "incident_id": i.incident_id,
                "incident_type": i.incident_type,
                "severity": i.severity,
                "title": i.title,
                "description": i.description,
                "status": i.status,
                "created_at": i.created_at.isoformat() if i.created_at else None,
            }
            for i in incidents
        ]

    async def get_prediction_calibration(
        self, days: int = 14
    ) -> dict[str, Any]:
        """Periksa kualitas kalibrasi prediksi model."""
        since = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=days)
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(Prediction).where(Prediction.timestamp >= since)
            )
            predictions = result.scalars().all()

        if not predictions:
            return {"sample_size": 0, "mean_probability": 0, "mean_confidence": 0}

        probs = [float(p.probability) for p in predictions]
        confs = [float(p.confidence) for p in predictions]
        return {
            "sample_size": len(predictions),
            "mean_probability": round(sum(probs) / len(probs), 4),
            "mean_confidence": round(sum(confs) / len(confs), 4),
            "model_versions": list({p.model_version for p in predictions}),
        }

    async def get_previous_proposals(
        self, strategy_version: str | None = None, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Ambil proposal Hermes sebelumnya untuk menghindari duplikasi."""
        async with AsyncSessionLocal() as db:
            stmt = select(Proposal).order_by(Proposal.created_at.desc()).limit(limit)
            if strategy_version:
                stmt = stmt.where(Proposal.strategy_version == strategy_version)
            result = await db.execute(stmt)
            proposals = result.scalars().all()
        return [
            {
                "proposal_id": p.proposal_id,
                "strategy_version": p.strategy_version,
                "problem_type": p.problem_type,
                "status": p.status,
                "proposed_change": p.proposed_change,
                "created_at": p.created_at.isoformat() if p.created_at else None,
            }
            for p in proposals
        ]


def _dossier_to_dict(d: TradeDossier) -> dict[str, Any]:
    return {
        "trade_id": d.trade_id,
        "strategy_version": d.strategy_version,
        "model_version": d.model_version,
        "config_version": d.config_version,
        "market_regime": d.market_regime,
        "entry_signal": d.entry_signal,
        "feature_snapshot": d.feature_snapshot,
        "risk_decision": d.risk_decision,
        "realized_pnl": float(d.realized_pnl),
        "exit_reason": d.exit_reason,
        "loss_classification": d.loss_classification,
        "technical_incidents": d.technical_incidents,
        "created_at": d.created_at.isoformat() if d.created_at else None,
        "closed_at": d.closed_at.isoformat() if d.closed_at else None,
    }
