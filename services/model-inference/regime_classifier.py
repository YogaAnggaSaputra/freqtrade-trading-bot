"""
regime_classifier.py
=====================
Unsupervised Market Regime Classifier menggunakan Gaussian Mixture Models (GMM)
dan Hidden Markov Models (HMM) sebagai fallback.

Menggantikan klasifikasi regime berbasis ADX statis dengan model statistik
yang mampu mendeteksi 5 regime tersembunyi (hidden states) secara otomatis:
  - sideways_low_vol   : Konsolidasi tenang, volume rendah
  - trending_up        : Uptrend dengan kekuatan ADX
  - trending_down      : Downtrend dengan kekuatan ADX
  - sideways_high_vol  : Konsolidasi volatile, choppy
  - breakout           : Ledakan volume & range, awal pergerakan besar

Model dilatih dari data candle historis dan di-persist ke /models/.
Jika tidak ada model, sistem fall back ke rule-based classifier.
"""
from __future__ import annotations

import logging
import os
import pickle
from typing import Any

import numpy as np

logger = logging.getLogger("model_inference.regime_classifier")

MODEL_DIR = os.getenv("MODEL_DIR", "/models")
GMM_MODEL_PATH = os.path.join(MODEL_DIR, "gmm_regime.pkl")
HMM_MODEL_PATH = os.path.join(MODEL_DIR, "hmm_regime.pkl")

N_REGIMES = 5

# Mapping regime index → label berdasarkan karakteristik cluster
REGIME_LABELS = [
    "sideways_low_vol",
    "trending_up",
    "trending_down",
    "sideways_high_vol",
    "breakout",
]


def _extract_regime_features(candles: list[dict[str, Any]]) -> np.ndarray:
    """
    Ekstrak feature vector untuk klasifikasi regime dari OHLCV.

    Features per bar (setelah candle ke-1):
      [0] log_return        : Return logaritmik harga close
      [1] hl_range_pct      : (High - Low) / Close — proxy volatilitas
      [2] volume_z          : Z-score volume dibanding rolling-20 window
      [3] close_position    : Posisi close di dalam bar (0=low, 1=high)
      [4] momentum_5        : Return kumulatif 5 bar terakhir
    """
    closes = np.array([float(c.get("close", 0)) for c in candles], dtype=float)
    highs = np.array([float(c.get("high", 0)) for c in candles], dtype=float)
    lows = np.array([float(c.get("low", 0)) for c in candles], dtype=float)
    volumes = np.array([float(c.get("volume", 0)) for c in candles], dtype=float)

    features = []
    vol_window = 20

    for i in range(1, len(closes)):
        if closes[i - 1] <= 0 or closes[i] <= 0:
            continue

        # Log return
        log_ret = np.log(closes[i] / closes[i - 1])

        # HL range as % of close
        hl_range = (highs[i] - lows[i]) / closes[i] if closes[i] > 0 else 0.0

        # Volume z-score
        vol_start = max(0, i - vol_window)
        vol_slice = volumes[vol_start:i]
        vol_mean = vol_slice.mean() if len(vol_slice) > 0 else volumes[i]
        vol_std = vol_slice.std() if len(vol_slice) > 1 else 1.0
        vol_z = (volumes[i] - vol_mean) / (vol_std + 1e-9)
        vol_z = float(np.clip(vol_z, -5, 5))

        # Close position within bar
        bar_range = highs[i] - lows[i]
        close_pos = (closes[i] - lows[i]) / bar_range if bar_range > 0 else 0.5

        # 5-bar momentum
        mom_start = max(0, i - 5)
        momentum = (closes[i] / closes[mom_start] - 1.0) if closes[mom_start] > 0 else 0.0

        features.append([log_ret, hl_range, vol_z, close_pos, momentum])

    return np.array(features, dtype=float) if features else np.zeros((1, 5))


class GMMRegimeClassifier:
    """
    Gaussian Mixture Model untuk klasifikasi regime pasar.
    Lebih ringan dari HMM dan cocok untuk CPU-only VPS.
    """

    def __init__(self, n_components: int = N_REGIMES, model_path: str = GMM_MODEL_PATH):
        self.n_components = n_components
        self.model_path = model_path
        self._gmm: Any | None = None
        self._scaler: Any | None = None
        self._regime_map: dict[int, str] = {}  # cluster idx → regime label
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    saved = pickle.load(f)
                self._gmm = saved["gmm"]
                self._scaler = saved.get("scaler")
                self._regime_map = saved.get("regime_map", {})
                logger.info("GMM regime model loaded from %s", self.model_path)
            except Exception as e:
                logger.warning("Failed to load GMM model: %s", e)

    def is_trained(self) -> bool:
        return self._gmm is not None

    def train(self, candles: list[dict[str, Any]]) -> dict[str, Any]:
        """Latih GMM dari data candle historis."""
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler

        if len(candles) < 100:
            return {"status": "insufficient_data", "samples": len(candles)}

        X = _extract_regime_features(candles)
        if X.shape[0] < 50:
            return {"status": "insufficient_features", "samples": X.shape[0]}

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type="full",
            max_iter=300,
            n_init=5,
            random_state=42,
        )
        gmm.fit(X_scaled)

        # Label setiap cluster berdasarkan karakteristik mean-nya
        regime_map = self._build_regime_map(gmm, scaler)

        self._gmm = gmm
        self._scaler = scaler
        self._regime_map = regime_map

        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump({"gmm": gmm, "scaler": scaler, "regime_map": regime_map}, f)

        logger.info("GMM trained. %d samples, BIC=%.1f", X.shape[0], gmm.bic(X_scaled))
        return {"status": "trained", "samples": int(X.shape[0]), "bic": round(float(gmm.bic(X_scaled)), 1)}

    def _build_regime_map(self, gmm: Any, scaler: Any) -> dict[int, str]:
        """
        Otomatis label setiap GMM cluster berdasarkan mean fitur:
          - mean log_return tinggi positif → trending_up
          - mean log_return tinggi negatif → trending_down
          - mean hl_range sangat tinggi & vol_z tinggi → breakout
          - mean hl_range sedang, vol_z tinggi → sideways_high_vol
          - sisanya → sideways_low_vol
        """
        means = scaler.inverse_transform(gmm.means_)
        # means columns: [log_ret, hl_range, vol_z, close_pos, momentum]
        regime_map: dict[int, str] = {}

        for i, m in enumerate(means):
            log_ret, hl_range, vol_z, close_pos, momentum = m
            if hl_range > 0.025 and vol_z > 1.5:
                label = "breakout"
            elif hl_range > 0.018 and vol_z > 0.5:
                label = "sideways_high_vol"
            elif momentum > 0.005 and log_ret > 0.001:
                label = "trending_up"
            elif momentum < -0.005 and log_ret < -0.001:
                label = "trending_down"
            else:
                label = "sideways_low_vol"
            regime_map[i] = label

        logger.info("GMM regime map: %s", regime_map)
        return regime_map

    def predict(self, candles: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Prediksi regime untuk candle terbaru.
        Returns dict dengan regime, state, confidence, dan metadata.
        """
        if not self.is_trained():
            return self._rule_based_fallback(candles)

        try:
            X = _extract_regime_features(candles)
            if X.shape[0] == 0:
                return self._rule_based_fallback(candles)

            # Gunakan 20 bar terakhir untuk prediksi
            X_recent = X[-20:] if X.shape[0] >= 20 else X

            X_scaled = self._scaler.transform(X_recent) if self._scaler else X_recent

            probs = self._gmm.predict_proba(X_scaled)
            # Rata-rata probabilitas dari beberapa bar terakhir untuk stabilitas
            mean_probs = probs.mean(axis=0)
            state = int(np.argmax(mean_probs))
            confidence = float(mean_probs[state])

            regime = self._regime_map.get(state, "sideways_low_vol")

            # Hitung metadata tambahan
            vol_pct = self._volatility_percentile(X)
            trend_strength = self._trend_strength(candles)
            duration = self._regime_duration(X_scaled)

            return {
                "regime": regime,
                "state": state,
                "confidence": round(confidence, 4),
                "volatility_percentile": round(vol_pct, 1),
                "trend_strength": round(trend_strength, 4),
                "regime_duration_bars": duration,
                "source": "gmm",
            }
        except Exception as e:
            logger.warning("GMM predict failed: %s — using rule-based", e)
            return self._rule_based_fallback(candles)

    def _volatility_percentile(self, features: np.ndarray) -> float:
        """Hitung persentil volatilitas dari hl_range (kolom 1)."""
        if features.shape[0] < 2:
            return 50.0
        hl_ranges = features[:, 1]  # hl_range_pct
        current = hl_ranges[-1]
        percentile = float(np.mean(hl_ranges <= current) * 100)
        return percentile

    def _trend_strength(self, candles: list[dict[str, Any]]) -> float:
        """Proxy trend strength dari momentum kumulatif."""
        if len(candles) < 14:
            return 0.0
        closes = [float(c.get("close", 0)) for c in candles[-14:]]
        if closes[0] <= 0:
            return 0.0
        total_return = abs(closes[-1] / closes[0] - 1.0)
        return float(min(total_return * 20, 1.0))  # Normalize ke [0, 1]

    def _regime_duration(self, X_scaled: np.ndarray) -> int:
        """Hitung berapa bar regime saat ini sudah berlangsung."""
        if not self.is_trained() or X_scaled.shape[0] < 2:
            return 1
        try:
            states = self._gmm.predict(X_scaled)
            current = states[-1]
            count = 1
            for s in reversed(states[:-1]):
                if s == current:
                    count += 1
                else:
                    break
            return count
        except Exception:
            return 1

    def _rule_based_fallback(self, candles: list[dict[str, Any]]) -> dict[str, Any]:
        """Fallback ke rule-based jika model belum dilatih."""
        if len(candles) < 2:
            return {
                "regime": "sideways_low_vol", "state": 0,
                "confidence": 0.3, "volatility_percentile": 50.0,
                "trend_strength": 0.0, "regime_duration_bars": 1, "source": "rule_based",
            }

        closes = [float(c.get("close", 0)) for c in candles[-20:] if c.get("close")]
        highs = [float(c.get("high", 0)) for c in candles[-20:] if c.get("high")]
        lows = [float(c.get("low", 0)) for c in candles[-20:] if c.get("low")]
        volumes = [float(c.get("volume", 0)) for c in candles[-20:] if c.get("volume")]

        if len(closes) < 5:
            return {
                "regime": "sideways_low_vol", "state": 0,
                "confidence": 0.3, "volatility_percentile": 50.0,
                "trend_strength": 0.0, "regime_duration_bars": 1, "source": "rule_based",
            }

        avg_range = np.mean([(h - l) / c for h, l, c in zip(highs, lows, closes, strict=False) if c > 0])
        momentum = (closes[-1] / closes[0] - 1.0) if closes[0] > 0 else 0.0
        avg_vol = np.mean(volumes[:-1]) if len(volumes) > 1 else volumes[0]
        vol_ratio = volumes[-1] / avg_vol if avg_vol > 0 else 1.0

        if avg_range > 0.025 and vol_ratio > 2.0:
            regime = "breakout"
        elif avg_range > 0.018:
            regime = "sideways_high_vol"
        elif momentum > 0.008:
            regime = "trending_up"
        elif momentum < -0.008:
            regime = "trending_down"
        else:
            regime = "sideways_low_vol"

        return {
            "regime": regime, "state": REGIME_LABELS.index(regime) if regime in REGIME_LABELS else 0,
            "confidence": 0.55, "volatility_percentile": float(avg_range * 2000),
            "trend_strength": float(min(abs(momentum) * 30, 1.0)),
            "regime_duration_bars": 1, "source": "rule_based_fallback",
        }
