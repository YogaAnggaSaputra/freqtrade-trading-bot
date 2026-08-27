"""
orchestrator.py
================
Experiment Orchestrator — mengelola seluruh pipeline pengujian kandidat strategi.

Pipeline: Backtest → Walk-forward → Stress Test → Monte Carlo → Shadow → Demo → Canary → Promote/Rollback

Orchestrator TIDAK mengubah strategi live secara otomatis.
Setiap promote ke live harus melalui approval manual dari Owner.
"""
from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from optuna_optimizer import OptunaOptimizer
from sqlalchemy import select

from shared.db.models import Experiment, Proposal
from shared.db.session import AsyncSessionLocal
from shared.messaging import Channels, MessageBus

logger = logging.getLogger("experiment_orchestrator.orchestrator")

# Status transitions yang valid
VALID_TRANSITIONS = {
    "pending": ["schema_validation", "rejected"],
    "schema_validation": ["backtest", "rejected"],
    "backtest": ["walk_forward", "failed"],
    "walk_forward": ["stress_test", "failed"],
    "stress_test": ["monte_carlo", "failed"],
    "monte_carlo": ["shadow", "failed"],
    "shadow": ["demo", "failed"],
    "demo": ["canary", "failed"],
    "canary": ["approved_for_promote", "failed", "rollback"],
    "approved_for_promote": ["promoted", "rollback"],
    "promoted": [],
    "failed": ["pending"],  # dapat diretry
    "rejected": [],
    "rollback": ["pending"],
}

# Threshold minimum agar eksperimen lolos setiap stage
PASS_THRESHOLDS = {
    "backtest": {
        "min_win_rate": 0.45,
        "max_drawdown": 0.12,
        "min_profit_factor": 1.2,
    },
    "walk_forward": {
        "min_win_rate": 0.43,
        "max_drawdown": 0.15,
        "min_profit_factor": 1.1,
    },
    "demo": {
        "min_days": 14,
        "max_drawdown_vs_baseline": 0.02,
    },
    "canary": {
        "max_drawdown_vs_baseline": 0.01,
        "min_days": 3,
    },
}


class ExperimentOrchestrator:
    """Mengelola lifecycle eksperimen dari proposal sampai promote/rollback."""

    def __init__(self, message_bus: MessageBus):
        self.bus = message_bus
        self.optuna = OptunaOptimizer()  # AutoML hyperparameter optimizer (background)

    async def accept_proposal(
        self,
        proposal_id: str,
        candidate_config: dict[str, Any],
        baseline_config: dict[str, Any],
    ) -> str | None:
        """
        Terima proposal yang lolos schema & policy validation.
        Buat eksperimen baru dengan status 'schema_validation'.
        """
        experiment_id = f"EXP-{uuid.uuid4().hex[:8].upper()}"

        async with AsyncSessionLocal() as db:
            # Periksa proposal ada dan statusnya pending
            result = await db.execute(
                select(Proposal).where(Proposal.proposal_id == proposal_id)
            )
            proposal = result.scalar_one_or_none()
            if not proposal or proposal.status != "pending":
                logger.warning("Proposal %s not found or not pending", proposal_id)
                return None

            experiment = Experiment(
                experiment_id=experiment_id,
                proposal_id=proposal_id,
                candidate_config=candidate_config,
                baseline_config=baseline_config,
                status="schema_validation",
                metrics={},
                started_at=datetime.now(UTC).replace(tzinfo=None),
            )
            db.add(experiment)
            proposal.status = "in_experiment"
            proposal.experiment_id = experiment_id
            await db.commit()

        logger.info("Created experiment: %s for proposal: %s", experiment_id, proposal_id)
        return experiment_id

    async def advance_stage(
        self,
        experiment_id: str,
        passed: bool,
        metrics: dict[str, Any],
        notes: str = "",
    ) -> str:
        """
        Maju ke stage berikutnya atau fail/rollback.
        Returns new status.
        """
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(Experiment).where(Experiment.experiment_id == experiment_id)
            )
            experiment = result.scalar_one_or_none()
            if not experiment:
                raise ValueError(f"Experiment {experiment_id} not found")

            current = str(experiment.status)  # type: ignore[arg-type]
            transitions = VALID_TRANSITIONS.get(current, [])

            if passed:
                next_stage = transitions[0] if transitions else current
            else:
                next_stage = "failed" if "failed" in transitions else "rejected"

            # Merge metrics
            existing_metrics = dict(experiment.metrics or {})  # type: ignore[arg-type]
            existing_metrics[current] = {
                "passed": passed,
                "metrics": metrics,
                "notes": notes,
                "completed_at": datetime.now(UTC).isoformat(),
            }

            experiment.status = next_stage
            experiment.metrics = existing_metrics
            if next_stage in ("promoted", "failed", "rejected", "rollback"):
                experiment.completed_at = datetime.now(UTC).replace(tzinfo=None)

            await db.commit()

        logger.info(
            "Experiment %s: %s -> %s (passed=%s)",
            experiment_id,
            current,
            next_stage,
            passed,
        )

        # Notifikasi via message bus
        await self.bus.publish(
            Channels.ALERT,
            {
                "type": "experiment_stage_update",
                "experiment_id": experiment_id,
                "from_stage": current,
                "to_stage": next_stage,
                "passed": passed,
                "metrics": metrics,
            },
        )
        return next_stage

    async def check_thresholds(
        self, stage: str, metrics: dict[str, Any]
    ) -> tuple[bool, str]:
        """
        Periksa apakah metrics memenuhi threshold untuk stage tertentu.
        Returns (passed, reason).
        """
        thresholds = PASS_THRESHOLDS.get(stage, {})
        if not thresholds:
            return True, f"No thresholds defined for stage {stage}"

        failures = []
        for key, limit in thresholds.items():
            actual = metrics.get(key)
            if actual is None:
                continue
            if key.startswith("min_") and actual < limit:
                failures.append(f"{key}: {actual} < {limit}")
            elif key.startswith("max_") and actual > limit:
                failures.append(f"{key}: {actual} > {limit}")

        if failures:
            return False, "; ".join(failures)
        return True, "All thresholds passed"

    async def get_pending_experiments(self) -> list[dict[str, Any]]:
        """Ambil semua eksperimen yang sedang berjalan."""
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(Experiment).where(
                    Experiment.status.not_in(["promoted", "failed", "rejected", "rollback"])
                ).order_by(Experiment.started_at)
            )
            experiments = result.scalars().all()

        return [
            {
                "experiment_id": e.experiment_id,
                "proposal_id": e.proposal_id,
                "status": e.status,
                "started_at": e.started_at.isoformat() if e.started_at else None,
                "metrics": e.metrics,
            }
            for e in experiments
        ]

    async def request_promotion(
        self,
        experiment_id: str,
        requested_by: str = "system",
    ) -> bool:
        """
        Request promote ke live — HARUS disetujui manual oleh Owner.
        Ini hanya mengubah status ke 'approved_for_promote', tidak langsung promote.
        """
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(Experiment).where(Experiment.experiment_id == experiment_id)
            )
            experiment = result.scalar_one_or_none()
            if not experiment or experiment.status != "canary":
                return False

            experiment.status = "approved_for_promote"
            await db.commit()

        await self.bus.publish(
            Channels.ALERT,
            {
                "type": "promotion_requested",
                "experiment_id": experiment_id,
                "requested_by": requested_by,
                "message": "Experiment passed canary. Manual Owner approval required to promote to live.",
            },
        )
        logger.info("Promotion requested for experiment: %s", experiment_id)
        return True

    async def run_optuna_optimization(
        self,
        historical_candles: list[dict[str, Any]],
        study_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Jalankan Optuna hyperparameter optimization di background.
        Dipanggil opsional setelah backtest selesai untuk mencari parameter terbaik.

        Args:
            historical_candles: Data candle historis untuk evaluasi
            study_name: Nama study, default auto-generated

        Returns:
            Hasil optimasi dengan best_params dan best_sharpe
        """
        logger.info("Triggering Optuna background optimization")
        result = await self.optuna.run_optimization(
            historical_candles=historical_candles,
            study_name=study_name,
        )
        if result.status == "completed":
            logger.info(
                "Optuna optimization complete: best Sharpe=%.3f, best_params=%s",
                result.best_value,
                result.best_params,
            )
            await self.bus.publish(
                Channels.ALERT,
                {
                    "type": "optuna_optimization_complete",
                    "study_name": result.study_name,
                    "best_sharpe": result.best_value,
                    "best_params": result.best_params,
                    "n_trials": result.n_trials,
                    "duration_seconds": result.duration_seconds,
                },
            )
        return result.to_dict()

    def get_optuna_status(self) -> dict[str, Any]:
        """Status Optuna optimizer saat ini."""
        last = self.optuna.get_last_result()
        return {
            "is_running": self.optuna.is_running,
            "last_result": last,
            "config": {
                "n_trials": int(__import__("os").getenv("OPTUNA_TRIALS", "50")),
                "sampler": __import__("os").getenv("OPTUNA_SAMPLER", "tpe"),
                "timeout_seconds": int(__import__("os").getenv("OPTUNA_TIMEOUT_SECONDS", "3600")),
            },
        }
