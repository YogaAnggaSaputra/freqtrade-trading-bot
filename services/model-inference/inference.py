"""
inference.py
=============
Model inference layer — memuat model ML dan menghasilkan probabilitas signal.
Mendukung: scikit-learn models (joblib), xgboost, dan rule-based fallback.
"""
from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger("model_inference.inference")

# Default model path
MODEL_DIR = os.getenv("MODEL_DIR", "/models")
ACTIVE_MODEL_VERSION = os.getenv("MODEL_VERSION", "v1_rule_based")


class ModelRegistry:
    """Registry model yang sudah disetujui — hanya load model dari sini."""

    _instance: ModelRegistry | None = None

    def __init__(self):
        self._models: dict[str, Any] = {}
        self._active_version: str = ACTIVE_MODEL_VERSION

    @classmethod
    def get_instance(cls) -> ModelRegistry:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_model(self, version: str) -> bool:
        """Muat model dari disk. Gagal gracefully jika tidak ada."""
        model_path = os.path.join(MODEL_DIR, f"{version}.joblib")
        if not os.path.exists(model_path):
            logger.warning("Model file not found: %s. Using rule-based fallback.", model_path)
            return False
        try:
            import joblib
            model = joblib.load(model_path)
            self._models[version] = model
            logger.info("Loaded model: %s", version)
            return True
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to load model %s: %s", version, e)
            return False

    def get_active_model(self) -> Any | None:
        return self._models.get(self._active_version)

    @property
    def active_version(self) -> str:
        return self._active_version


class InferenceEngine:
    """
    Mesin inferensi yang menggabungkan fitur teknikal dengan model ML.
    Jika model tidak tersedia, gunakan rule-based fallback.
    """

    def __init__(self):
        self.registry = ModelRegistry.get_instance()
        self._attempted_load = False

    def _ensure_model_loaded(self):
        if not self._attempted_load:
            self.registry.load_model(self.registry.active_version)
            self._attempted_load = True

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        """
        Hasilkan prediksi berdasarkan fitur.

        Returns:
            Dict dengan:
            - probability: float [0, 1] probabilitas arah bullish
            - confidence: float [0, 1]
            - regime: str
            - signal: str (buy / sell / hold)
            - model_version: str
            - timestamp: str ISO
        """
        self._ensure_model_loaded()
        model = self.registry.get_active_model()
        regime = features.get("regime", "unknown")

        if model is not None:
            return self._ml_predict(model, features, regime)
        else:
            return self._rule_based_predict(features, regime)

    def _ml_predict(
        self, model: Any, features: dict[str, Any], regime: str
    ) -> dict[str, Any]:
        """Gunakan model scikit-learn / XGBoost untuk prediksi."""
        try:
            feature_keys = [
                "ema_cross_9_21", "ema_cross_21_50",
                "rsi_14", "atr_pct", "adx_14",
                "bb_width", "bb_position",
                "volume_zscore", "price_position_20",
                "roc_5", "roc_10",
            ]
            import numpy as np
            X = np.array([[features.get(k, 0.0) for k in feature_keys]])
            probability = float(model.predict_proba(X)[0][1])
            confidence = min(abs(probability - 0.5) * 2.0 + 0.5, 1.0)
            signal = self._probability_to_signal(probability, confidence, regime)
            return {
                "probability": round(probability, 4),
                "confidence": round(confidence, 4),
                "regime": regime,
                "signal": signal,
                "model_version": self.registry.active_version,
                "timestamp": datetime.now(UTC).isoformat(),
                "source": "ml_model",
            }
        except Exception as e:  # noqa: BLE001
            logger.warning("ML prediction failed: %s. Falling back to rule-based.", e)
            return self._rule_based_predict(features, regime)

    def _rule_based_predict(
        self, features: dict[str, Any], regime: str
    ) -> dict[str, Any]:
        """
        Fallback rule-based prediction untuk saat model tidak tersedia.
        Menggunakan EMA crossover + RSI + ADX.
        """
        ema_cross = features.get("ema_cross_9_21", 0.0)
        rsi = features.get("rsi_14", 50.0)
        adx = features.get("adx_14", 15.0)
        bb_pos = features.get("bb_position", 0.5)

        # Hitung score sederhana [-1, 1]
        score = 0.0
        if ema_cross > 0:
            score += 0.3
        elif ema_cross < 0:
            score -= 0.3

        if rsi < 35:
            score += 0.2
        elif rsi > 65:
            score -= 0.2

        if adx > 25:
            score += 0.15 if ema_cross > 0 else -0.15

        if bb_pos < 0.2:
            score += 0.1
        elif bb_pos > 0.8:
            score -= 0.1

        # Normalize ke [0, 1]
        probability = max(0.0, min(1.0, (score + 1.0) / 2.0))
        confidence = min(abs(score) + 0.3, 1.0)

        # Di regime sideways, turunkan confidence
        if "sideways" in regime:
            confidence *= 0.7
        # Di breakout, naikkan confidence
        elif "breakout" in regime:
            confidence = min(confidence * 1.2, 1.0)

        signal = self._probability_to_signal(probability, confidence, regime)

        return {
            "probability": round(probability, 4),
            "confidence": round(confidence, 4),
            "regime": regime,
            "signal": signal,
            "model_version": "rule_based_fallback",
            "timestamp": datetime.now(UTC).isoformat(),
            "source": "rule_based",
        }

    def _probability_to_signal(
        self, probability: float, confidence: float, regime: str
    ) -> str:
        """
        Konversi probabilitas ke signal trading.
        Di regime sideways, hanya izinkan signal dengan confidence sangat tinggi.
        """
        min_confidence = 0.70
        if "sideways" in regime:
            min_confidence = 0.85

        if confidence < min_confidence:
            return "hold"
        if probability >= 0.60:
            return "buy"
        if probability <= 0.40:
            return "sell"
        return "hold"
