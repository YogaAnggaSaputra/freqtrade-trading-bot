"""
proposal_generator.py
=======================
Menghasilkan proposal eksperimen tervalidasi dari analisis loss.
Output berupa JSON schema yang harus lolos validasi policy sebelum masuk Experiment Orchestrator.

Hermes HANYA membuat proposal — tidak pernah langsung mengubah config atau strategi.
"""
from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger("hermes.proposal_generator")

# Tipe problem yang dikenali Hermes
KNOWN_PROBLEM_TYPES = {
    "regime_mismatch",        # Strategi tidak cocok untuk regime saat ini
    "entry_timing_poor",      # Entry terlalu early/late
    "stop_loss_too_tight",    # SL terkena noise sebelum profit
    "stop_loss_too_wide",     # SL terlalu longgar, loss terlalu besar
    "feature_drift",          # Fitur bergeser dari distribusi training
    "model_calibration",      # Prediksi probability tidak terkalibrasi
    "parameter_staleness",    # Parameter sudah usang, pasar berubah
    "execution_issues",       # Masalah teknis: latency, slippage tinggi
    "risk_param_conservative",# Risk terlalu konservatif, banyak missed
}

# Kelas perubahan yang diizinkan (tidak boleh menyentuh leverage, drawdown limit, API key)
SAFE_CHANGE_CLASSES = {"safe_experiment", "parameter_tune", "feature_update"}
UNSAFE_CHANGE_CLASSES = {"leverage_change", "drawdown_limit_change", "api_key_change"}


class ProposedChange(BaseModel):
    change_class: str = Field(..., description="Kelas perubahan (safe_experiment, parameter_tune, dll)")
    parameter: str = Field(..., description="Nama parameter yang diubah")
    old_value: Any = Field(..., description="Nilai lama")
    new_value: Any = Field(..., description="Nilai baru yang diusulkan")
    rationale: str = Field(default="", description="Alasan perubahan")

    @model_validator(mode="after")
    def validate_change_class(self) -> ProposedChange:
        if self.change_class in UNSAFE_CHANGE_CLASSES:
            raise ValueError(
                f"Perubahan kelas '{self.change_class}' tidak diizinkan. "
                f"Hanya {SAFE_CHANGE_CLASSES} yang dapat diusulkan."
            )
        return self


class HermesProposal(BaseModel):
    proposal_id: str = Field(default_factory=lambda: f"HERMES-{datetime.now(UTC).strftime('%Y-%m-%d')}-{uuid.uuid4().hex[:6].upper()}")
    strategy_version: str
    problem_type: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    proposed_change: ProposedChange
    expected_effect: str = ""
    validation_plan: str = "backtest_multi_period_then_demo_14d"
    rollback_condition: str = "canary_drawdown_gt_baseline_by_1_percent"
    status: str = "pending"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def validate_problem_type(self) -> HermesProposal:
        if self.problem_type not in KNOWN_PROBLEM_TYPES:
            logger.warning(
                "Unknown problem type: %s. Proposal akan tetap dibuat.",
                self.problem_type,
            )
        return self

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "strategy_version": self.strategy_version,
            "problem_type": self.problem_type,
            "evidence": self.evidence,
            "proposed_change": {
                "class": self.proposed_change.change_class,
                "parameter": self.proposed_change.parameter,
                "old_value": self.proposed_change.old_value,
                "new_value": self.proposed_change.new_value,
                "rationale": self.proposed_change.rationale,
            },
            "expected_effect": self.expected_effect,
            "validation_plan": self.validation_plan,
            "rollback_condition": self.rollback_condition,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
        }


class ProposalGenerator:
    """
    Menghasilkan proposal eksperimen dari analisis evidence.
    Setiap rule mengembalikan Optional[HermesProposal].
    """

    def __init__(self, strategy_version: str):
        self.strategy_version = strategy_version

    def generate_all(
        self,
        loss_summary: dict[str, Any],
        regime_performance: dict[str, Any],
        incidents: list[dict[str, Any]],
        calibration: dict[str, Any],
        previous_proposals: list[dict[str, Any]],
    ) -> list[HermesProposal]:
        """Jalankan semua rule dan kumpulkan proposal yang relevan."""
        proposals: list[HermesProposal] = []

        rules = [
            self._check_regime_mismatch,
            self._check_stop_loss_too_tight,
            self._check_model_calibration_drift,
            self._check_high_loss_streak,
            self._check_execution_issues,
        ]

        already_proposed = {p.get("problem_type") for p in previous_proposals if p.get("status") == "pending"}

        for rule in rules:
            try:
                proposal = rule(loss_summary, regime_performance, incidents, calibration)
                if proposal and proposal.problem_type not in already_proposed:
                    proposals.append(proposal)
            except Exception as e:  # noqa: BLE001
                logger.warning("Rule %s failed: %s", rule.__name__, e)

        logger.info("Generated %d proposal(s)", len(proposals))
        return proposals

    def _check_regime_mismatch(
        self,
        loss_summary: dict[str, Any],
        regime_performance: dict[str, Any],
        _incidents: list,
        _calibration: dict,
    ) -> HermesProposal | None:
        """Deteksi bila satu regime secara konsisten mengalami net loss."""
        for regime, stats in regime_performance.items():
            if stats.get("sample_size", 0) < 10:
                continue
            net_pnl = stats.get("net_pnl", 0)
            win_rate = stats.get("win_rate", 1.0)
            if net_pnl < -0.02 and win_rate < 0.40:
                return HermesProposal(
                    strategy_version=self.strategy_version,
                    problem_type="regime_mismatch",
                    evidence={
                        "regime": regime,
                        "sample_size": stats["sample_size"],
                        "net_pnl": net_pnl,
                        "win_rate": win_rate,
                    },
                    proposed_change=ProposedChange(
                        change_class="safe_experiment",
                        parameter="adx_min_threshold",
                        old_value=15,
                        new_value=20,
                        rationale=f"Naikkan ADX threshold untuk kurangi entry di regime {regime}",
                    ),
                    expected_effect=f"Kurangi jumlah entry berkualitas rendah di regime {regime}",
                    validation_plan="walk_forward_6_windows_then_demo_14d",
                    rollback_condition="canary_drawdown_gt_baseline_by_1_percent",
                )
        return None

    def _check_stop_loss_too_tight(
        self,
        loss_summary: dict[str, Any],
        _regime: dict,
        _incidents: list,
        _calibration: dict,
    ) -> HermesProposal | None:
        """Deteksi bila banyak trade di-stop sebelum profit (SL terlalu tight)."""
        exit_reasons = loss_summary.get("exit_reason_breakdown", {})
        sl_hits = exit_reasons.get("stop_loss", 0) + exit_reasons.get("stoploss", 0)
        total = loss_summary.get("sample_size", 0)
        if total < 20 or sl_hits == 0:
            return None
        sl_rate = sl_hits / total
        if sl_rate > 0.60:  # >60% trade kena SL
            return HermesProposal(
                strategy_version=self.strategy_version,
                problem_type="stop_loss_too_tight",
                evidence={
                    "sample_size": total,
                    "sl_hit_count": sl_hits,
                    "sl_hit_rate": round(sl_rate, 4),
                },
                proposed_change=ProposedChange(
                    change_class="parameter_tune",
                    parameter="stoploss_atr_multiplier",
                    old_value=1.5,
                    new_value=2.0,
                    rationale="Naikkan SL multiplier untuk beri lebih banyak ruang gerak",
                ),
                expected_effect="Kurangi premature stop-out, tingkatkan win rate",
                validation_plan="backtest_12_months_then_walk_forward_4_windows",
                rollback_condition="canary_avg_loss_gt_1_5x_baseline",
            )
        return None

    def _check_model_calibration_drift(
        self,
        _loss: dict,
        _regime: dict,
        _incidents: list,
        calibration: dict[str, Any],
    ) -> HermesProposal | None:
        """Deteksi bila confidence model sangat tinggi tapi win rate rendah."""
        mean_conf = calibration.get("mean_confidence", 0)
        sample = calibration.get("sample_size", 0)
        if sample < 50 or mean_conf <= 0:
            return None
        # Bila model sangat confident (>0.8) tapi kita tahu dari loss summary banyak loss
        # (ini diuji di integrasi dengan loss_summary di agent.py)
        if mean_conf > 0.85:
            return HermesProposal(
                strategy_version=self.strategy_version,
                problem_type="model_calibration",
                evidence={
                    "sample_size": sample,
                    "mean_confidence": mean_conf,
                },
                proposed_change=ProposedChange(
                    change_class="feature_update",
                    parameter="confidence_threshold",
                    old_value=0.60,
                    new_value=0.70,
                    rationale="Naikkan threshold confidence untuk filter sinyal berkualitas rendah",
                ),
                expected_effect="Kurangi false positive dari model yang terlalu yakin",
                validation_plan="shadow_mode_7d_then_demo_14d",
                rollback_condition="signal_frequency_drops_below_1_per_day",
            )
        return None

    def _check_high_loss_streak(
        self,
        loss_summary: dict[str, Any],
        _regime: dict,
        _incidents: list,
        _calibration: dict,
    ) -> HermesProposal | None:
        """Deteksi loss streak panjang yang menunjukkan masalah sistematis."""
        loss_streak = loss_summary.get("loss_streak", 0)
        if loss_streak >= 8:
            return HermesProposal(
                strategy_version=self.strategy_version,
                problem_type="parameter_staleness",
                evidence={
                    "loss_streak": loss_streak,
                    "net_pnl": loss_summary.get("net_pnl", 0),
                    "win_rate": loss_summary.get("win_rate", 0),
                },
                proposed_change=ProposedChange(
                    change_class="safe_experiment",
                    parameter="entry_signal_lookback_period",
                    old_value=14,
                    new_value=10,
                    rationale="Kurangi lookback agar lebih responsif terhadap perubahan pasar",
                ),
                expected_effect="Adaptasi lebih cepat terhadap perubahan regime pasar",
                validation_plan="backtest_6_months_then_shadow_7d",
                rollback_condition="loss_streak_exceeds_5_in_canary",
            )
        return None

    def _check_execution_issues(
        self,
        _loss: dict,
        _regime: dict,
        incidents: list[dict[str, Any]],
        _calibration: dict,
    ) -> HermesProposal | None:
        """Deteksi insiden teknis berulang yang mempengaruhi eksekusi."""
        execution_incidents = [
            i for i in incidents
            if i.get("incident_type") in ("api_timeout", "order_mismatch", "fill_slippage_high")
            and i.get("severity") in ("high", "critical")
        ]
        if len(execution_incidents) >= 3:
            return HermesProposal(
                strategy_version=self.strategy_version,
                problem_type="execution_issues",
                evidence={
                    "incident_count": len(execution_incidents),
                    "incident_types": list({i["incident_type"] for i in execution_incidents}),
                },
                proposed_change=ProposedChange(
                    change_class="parameter_tune",
                    parameter="order_timeout_seconds",
                    old_value=10,
                    new_value=20,
                    rationale="Naikkan timeout order untuk mengurangi false failure",
                ),
                expected_effect="Kurangi false order timeout dan reconciliation mismatch",
                validation_plan="shadow_mode_3d_then_demo_7d",
                rollback_condition="execution_latency_gt_5s_median",
            )
        return None
