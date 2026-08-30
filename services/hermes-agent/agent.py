"""
agent.py
=========
Hermes Improvement Agent — otak dari sistem adaptive trading.

Tugas Hermes:
1. Kumpulkan evidence dari data trading (read-only)
2. Analisis pola loss dan drift
3. Hasilkan proposal eksperimen tervalidasi
4. Kirim proposal ke Policy Gate via message bus

Hermes TIDAK PERNAH:
- Mengakses API key
- Mengirim order langsung
- Mengubah config live
- Mempromosikan strategi tanpa approval manual
"""
from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from decay_detector import DecayDetector
from evidence_collector import EvidenceCollector
from proposal_generator import HermesProposal, ProposalGenerator
from sqlalchemy import and_, select
from shared.quant.supreme_final import ThompsonProposalSelector

from shared.db.models import Proposal
from shared.db.session import AsyncSessionLocal
from shared.messaging import Channels, MessageBus

logger = logging.getLogger("hermes.agent")


class HermesAgent:
    """
    Hermes Autonomous Improvement Agent.
    Berjalan dalam loop periodik, menganalisis data, dan menghasilkan proposal.
    """

    def __init__(
        self,
        message_bus: MessageBus,
        analysis_interval_hours: int = 6,
        lookback_days: int = 14,
        min_samples: int = 20,
    ):
        self.bus = message_bus
        self.analysis_interval = analysis_interval_hours * 3600
        self.lookback_days = lookback_days
        self.min_samples = min_samples
        self.collector = EvidenceCollector()
        self.decay_detector = DecayDetector()  # Strategy decay & concept drift detector
        self.mab_selector = ThompsonProposalSelector([
            "parameter_tune", "risk_reduction", "pair_whitelist_update", "regime_threshold_adj"
        ])
        self._running = False
        self._last_decay_report: dict = {}

    async def start(self) -> None:
        """Jalankan agent dalam loop periodik."""
        self._running = True
        logger.info(
            "Hermes Agent started. Analysis interval: %dh, Lookback: %dd",
            self.analysis_interval // 3600,
            self.lookback_days,
        )
        while self._running:
            try:
                await self.run_analysis_cycle()
            except Exception as e:  # noqa: BLE001
                logger.error("Analysis cycle failed: %s", e, exc_info=True)
            await asyncio.sleep(self.analysis_interval)

    async def stop(self) -> None:
        self._running = False
        logger.info("Hermes Agent stopped.")

    async def run_analysis_cycle(self) -> list[dict[str, Any]]:
        """
        Satu siklus analisis lengkap.
        Returns list proposal yang dihasilkan dan disimpan.
        """
        logger.info("=== Hermes Analysis Cycle Started ===")
        cycle_start = datetime.now(UTC)

        # 1. Tentukan versi strategi aktif
        strategy_version = await self.collector.get_active_strategy_version()
        if not strategy_version:
            logger.warning("No active strategy deployment found. Skipping cycle.")
            return []

        # 2. Kumpulkan semua evidence (read-only)
        logger.info("Collecting evidence for strategy: %s", strategy_version)

        loss_summary = await self.collector.get_loss_summary(
            days=self.lookback_days,
            strategy_version=strategy_version,
        )

        if loss_summary.get("sample_size", 0) < self.min_samples:
            logger.info(
                "Insufficient samples: %d < %d. Waiting for more data.",
                loss_summary.get("sample_size", 0),
                self.min_samples,
            )
            return []

        regime_performance = await self.collector.get_regime_performance(
            days=self.lookback_days
        )
        incidents = await self.collector.get_recent_incidents(days=7)
        calibration = await self.collector.get_prediction_calibration(
            days=self.lookback_days
        )
        previous_proposals = await self.collector.get_previous_proposals(
            strategy_version=strategy_version, limit=20
        )

        # 3. Generate proposals from loss analysis
        generator = ProposalGenerator(strategy_version=strategy_version)
        proposals = generator.generate_all(
            loss_summary=loss_summary,
            regime_performance=regime_performance,
            incidents=incidents,
            calibration=calibration,
            previous_proposals=previous_proposals,
        )

        # 4. Strategy Decay Detection (additional independent check)
        try:
            decay_report = await self.decay_detector.analyze(strategy_version)
            self._last_decay_report = decay_report.to_dict()
            if decay_report.decay_detected:
                logger.warning(
                    "Strategy decay detected! Type: %s, Severity: %s",
                    decay_report.decay_type, decay_report.severity,
                )
                # Publish decay alert to Redis/Telegram
                await self.bus.publish(Channels.ALERT, {
                    "type": "strategy_decay_alert",
                    "strategy_version": strategy_version,
                    "decay_type": decay_report.decay_type,
                    "severity": decay_report.severity,
                    "recommendation": decay_report.recommendation,
                    "auto_retrain_suggested": decay_report.auto_retrain_suggested,
                    "details": decay_report.details,
                })
        except Exception as e:  # noqa: BLE001
            logger.warning("Decay detection failed (non-fatal): %s", e)

        if not proposals:
            logger.info("No proposals generated this cycle.")
            return []

        # 4. Persist proposals ke database
        saved_proposals = []
        for proposal in proposals:
            try:
                saved = await self._save_proposal(proposal)
                if saved:
                    saved_proposals.append(proposal.to_json_dict())
                    # 5. Publish ke message bus untuk Policy Gate
                    await self.bus.publish(
                        Channels.ALERT,
                        {
                            "type": "hermes_proposal",
                            "proposal_id": proposal.proposal_id,
                            "problem_type": proposal.problem_type,
                            "strategy_version": proposal.strategy_version,
                            "created_at": proposal.created_at.isoformat(),
                        },
                    )
            except Exception as e:  # noqa: BLE001
                logger.error("Failed to save/publish proposal: %s", e)

        cycle_duration = (datetime.now(UTC) - cycle_start).total_seconds()
        logger.info(
            "=== Hermes Cycle Complete: %d proposals in %.1fs ===",
            len(saved_proposals),
            cycle_duration,
        )
        return saved_proposals

    async def _save_proposal(self, proposal: HermesProposal) -> bool:
        """Simpan proposal ke database."""
        async with AsyncSessionLocal() as db:
            # Cek duplikat
            existing = await db.execute(
                select(Proposal).where(
                    and_(
                        Proposal.strategy_version == proposal.strategy_version,
                        Proposal.problem_type == proposal.problem_type,
                        Proposal.status == "pending",
                    )
                )
            )
            if existing.scalar_one_or_none():
                logger.info(
                    "Proposal untuk %s / %s sudah pending. Skip.",
                    proposal.strategy_version,
                    proposal.problem_type,
                )
                return False

            db_proposal = Proposal(
                proposal_id=proposal.proposal_id,
                strategy_version=proposal.strategy_version,
                problem_type=proposal.problem_type,
                evidence=proposal.evidence,
                proposed_change=proposal.proposed_change.model_dump(),
                expected_effect=proposal.expected_effect,
                validation_plan=proposal.validation_plan,
                rollback_condition=proposal.rollback_condition,
                status="pending",
                created_at=proposal.created_at.replace(tzinfo=None),
                updated_at=proposal.created_at.replace(tzinfo=None),
            )
            db.add(db_proposal)
            await db.commit()
            logger.info("Saved proposal: %s", proposal.proposal_id)
            return True
