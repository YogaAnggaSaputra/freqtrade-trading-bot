"""
uncertainty_filter.py
======================
Conformal Prediction & Uncertainty-Based Trade Filter

Memblokir eksekusi trade ketika model ML tidak yakin dengan prediksinya.
Ketidakpastian diukur dengan dua dimensi:
  1. std_dev: Seberapa berbeda prediksi antar base models (model disagreement)
  2. entropy : Seberapa "biner" prediksi — mendekati 0.5 = sangat tidak yakin

Filosofi:
  "Lebih baik tidak trade daripada trade dengan model yang bingung."
  Ketika pasar sedang dalam kondisi yang tidak familiar untuk model,
  lebih baik skip trade dan simpan modal.

Thresholds (configurable via env vars):
  UNCERTAINTY_MIN_CONFIDENCE=0.70   → Probability harus >= 70% dari batas
  UNCERTAINTY_MAX_STD_DEV=0.20      → Max 20% std dev antar base models
  UNCERTAINTY_MAX_ENTROPY=0.65      → Max entropy 65% (lebih rendah = lebih yakin)
  UNCERTAINTY_ENABLED=true

Mode degraded (jika ensemble belum dilatih):
  Hanya pakai threshold confidence dari single model (tidak ada std_dev check)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from shared.quant.calibration import platt_apply, expected_calibration_error

logger = logging.getLogger("model_inference.uncertainty_filter")

# ── Configuration ──────────────────────────────────────────────────────────────
UNCERTAINTY_ENABLED         = os.getenv("UNCERTAINTY_ENABLED", "true").lower() == "true"
UNCERTAINTY_MIN_CONFIDENCE  = float(os.getenv("UNCERTAINTY_MIN_CONFIDENCE", "0.70"))
UNCERTAINTY_MAX_STD_DEV     = float(os.getenv("UNCERTAINTY_MAX_STD_DEV", "0.20"))
UNCERTAINTY_MAX_ENTROPY     = float(os.getenv("UNCERTAINTY_MAX_ENTROPY", "0.65"))


@dataclass
class UncertaintyResult:
    """Hasil pengecekan uncertainty filter dengan probabilitas terkalibrasi."""
    should_trade: bool
    confidence_ok: bool
    std_dev_ok: bool
    entropy_ok: bool
    probability: float
    calibrated_probability: float
    std_dev: float
    entropy: float
    reason: str
    details: dict[str, Any]


class UncertaintyFilter:
    """
    Filter trade berdasarkan tingkat kepastian model ML dan kalibrasi probabilitas.
    Diintegrasikan ke dalam inference pipeline setelah ensemble prediction.
    """

    @staticmethod
    def check(
        probability: float,
        std_dev: float = 0.0,
        entropy: float = 0.0,
        ensemble_available: bool = False,
        platt_model: dict | None = None,
    ) -> UncertaintyResult:
        """
        Cek apakah prediksi cukup pasti untuk dieksekusi.
        Jika platt_model diberikan, probability dikalibrasi terlebih dahulu.
        """
        calibrated_p = platt_apply(probability, platt_model) if platt_model else probability
        eval_p = calibrated_p

        if not UNCERTAINTY_ENABLED:
            return UncertaintyResult(
                should_trade=True,
                confidence_ok=True,
                std_dev_ok=True,
                entropy_ok=True,
                probability=probability,
                calibrated_probability=eval_p,
                std_dev=std_dev,
                entropy=entropy,
                reason="Uncertainty filter disabled",
                details={"mode": "disabled"},
            )

        reasons_fail = []

        # ── Check 1: Calibrated minimum confidence / probability ──────────────
        confidence_ok = (eval_p >= UNCERTAINTY_MIN_CONFIDENCE) or (eval_p <= 1 - UNCERTAINTY_MIN_CONFIDENCE)
        if not confidence_ok:
            reasons_fail.append(
                f"Low confidence: calibrated_prob={eval_p:.3f} (raw={probability:.3f}) "
                f"not meeting threshold ({UNCERTAINTY_MIN_CONFIDENCE:.0%} or {1-UNCERTAINTY_MIN_CONFIDENCE:.0%})"
            )

        # ── Check 2: Model disagreement (std_dev) — only if ensemble available ─
        std_dev_ok = True
        if ensemble_available and std_dev > 0:
            std_dev_ok = std_dev <= UNCERTAINTY_MAX_STD_DEV
            if not std_dev_ok:
                reasons_fail.append(
                    f"High model disagreement: std_dev={std_dev:.3f} > {UNCERTAINTY_MAX_STD_DEV}"
                )

        # ── Check 3: Entropy (distributional uncertainty) ─────────────────────
        entropy_ok = True
        if ensemble_available and entropy > 0:
            entropy_ok = entropy <= UNCERTAINTY_MAX_ENTROPY
            if not entropy_ok:
                reasons_fail.append(
                    f"High prediction entropy: {entropy:.3f} > {UNCERTAINTY_MAX_ENTROPY} — market in uncertain state"
                )

        should_trade = confidence_ok and std_dev_ok and entropy_ok

        if should_trade:
            reason = (
                f"Prediction confident: cal_p={eval_p:.3f} (raw={probability:.3f}), "
                f"std={std_dev:.3f}, entropy={entropy:.3f}"
            )
        else:
            reason = "BLOCKED: " + " | ".join(reasons_fail)

        details = {
            "raw_probability": probability,
            "calibrated_probability": eval_p,
            "std_dev": std_dev,
            "entropy": entropy,
            "conviction": abs(eval_p - 0.5) * 2,
            "ensemble_available": ensemble_available,
            "platt_calibrated": platt_model is not None,
            "thresholds": {
                "min_confidence": UNCERTAINTY_MIN_CONFIDENCE,
                "max_std_dev": UNCERTAINTY_MAX_STD_DEV,
                "max_entropy": UNCERTAINTY_MAX_ENTROPY,
            },
        }

        return UncertaintyResult(
            should_trade=should_trade,
            confidence_ok=confidence_ok,
            std_dev_ok=std_dev_ok,
            entropy_ok=entropy_ok,
            probability=probability,
            calibrated_probability=eval_p,
            std_dev=std_dev,
            entropy=entropy,
            reason=reason,
            details=details,
        )

    @staticmethod
    def check_from_ensemble_result(ensemble_result: dict[str, Any]) -> UncertaintyResult:
        """
        Shortcut: check langsung dari dict hasil ensemble.predict_proba_with_uncertainty().
        """
        return UncertaintyFilter.check(
            probability=ensemble_result.get("probability", 0.5),
            std_dev=ensemble_result.get("std_dev", 0.0),
            entropy=ensemble_result.get("entropy", 0.0),
            ensemble_available=ensemble_result.get("ensemble_available", False),
        )
