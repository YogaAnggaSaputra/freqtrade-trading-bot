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

logger = logging.getLogger("model_inference.uncertainty_filter")

# ── Configuration ──────────────────────────────────────────────────────────────
UNCERTAINTY_ENABLED         = os.getenv("UNCERTAINTY_ENABLED", "true").lower() == "true"
UNCERTAINTY_MIN_CONFIDENCE  = float(os.getenv("UNCERTAINTY_MIN_CONFIDENCE", "0.70"))
UNCERTAINTY_MAX_STD_DEV     = float(os.getenv("UNCERTAINTY_MAX_STD_DEV", "0.20"))
UNCERTAINTY_MAX_ENTROPY     = float(os.getenv("UNCERTAINTY_MAX_ENTROPY", "0.65"))


@dataclass
class UncertaintyResult:
    """Hasil pengecekan uncertainty filter."""
    should_trade: bool
    confidence_ok: bool
    std_dev_ok: bool
    entropy_ok: bool
    probability: float
    std_dev: float
    entropy: float
    reason: str
    details: dict[str, Any]


class UncertaintyFilter:
    """
    Filter trade berdasarkan tingkat kepastian model ML.
    Diintegrasikan ke dalam inference pipeline setelah ensemble prediction.
    """

    @staticmethod
    def check(
        probability: float,
        std_dev: float = 0.0,
        entropy: float = 0.0,
        ensemble_available: bool = False,
    ) -> UncertaintyResult:
        """
        Cek apakah prediksi cukup pasti untuk dieksekusi.

        Args:
            probability  : Final probability dari model (0.0–1.0)
            std_dev      : Std deviation antar base models (0.0–0.5)
            entropy      : Shannon entropy prediksi (0.0–1.0)
            ensemble_available: True jika ensemble sudah dilatih

        Returns:
            UncertaintyResult dengan should_trade=True jika aman untuk trade
        """
        if not UNCERTAINTY_ENABLED:
            return UncertaintyResult(
                should_trade=True,
                confidence_ok=True,
                std_dev_ok=True,
                entropy_ok=True,
                probability=probability,
                std_dev=std_dev,
                entropy=entropy,
                reason="Uncertainty filter disabled",
                details={"mode": "disabled"},
            )

        reasons_fail = []

        # ── Check 1: Minimum confidence / probability ──────────────────────────
        # Probability harus > threshold (untuk Long) atau < 1-threshold (untuk Short)
        # Kita cek dari "conviction strength": seberapa jauh dari 0.5
        conviction = abs(probability - 0.5) * 2  # 0.0 (no conviction) → 1.0 (full conviction)
        confidence_ok = conviction >= (UNCERTAINTY_MIN_CONFIDENCE - 0.5) * 2

        # Simplified: probability > threshold ATAU < (1 - threshold)
        confidence_ok = (probability >= UNCERTAINTY_MIN_CONFIDENCE) or (probability <= 1 - UNCERTAINTY_MIN_CONFIDENCE)
        if not confidence_ok:
            reasons_fail.append(
                f"Low confidence: probability={probability:.3f} not meeting "
                f"threshold ({UNCERTAINTY_MIN_CONFIDENCE:.0%} or {1-UNCERTAINTY_MIN_CONFIDENCE:.0%})"
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
                f"Prediction confident: p={probability:.3f}, "
                f"std={std_dev:.3f}, entropy={entropy:.3f}"
            )
        else:
            reason = "BLOCKED: " + " | ".join(reasons_fail)

        details = {
            "probability": probability,
            "std_dev": std_dev,
            "entropy": entropy,
            "conviction": abs(probability - 0.5) * 2,
            "ensemble_available": ensemble_available,
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
