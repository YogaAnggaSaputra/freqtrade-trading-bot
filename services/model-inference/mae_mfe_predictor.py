"""
mae_mfe_predictor.py
=====================
ML-Based Stop-Loss & Take-Profit Optimizer menggunakan Gradient Boosting.

Memperkirakan:
  - MAE (Maximum Adverse Excursion): seberapa jauh harga bergerak melawan posisi
    sebelum akhirnya berbalik profit — digunakan untuk menentukan Stop-Loss optimal.
  - MFE (Maximum Favorable Excursion): seberapa jauh harga bergerak menguntungkan
    posisi — digunakan untuk menentukan Take-Profit yang realistis.

Model dilatih dari data TradeDossier historis yang sudah closed (real trade).
Dengan model ini, SL/TP tidak lagi bergantung pada multiplier ATR statis
tapi disesuaikan secara dinamis berdasarkan kondisi pasar saat ini.

Integrasi:
  - /mae-mfe-predict endpoint di model-inference
  - risk-gateway memanggil endpoint ini sebelum mengirim order ke Binance
"""
from __future__ import annotations

import logging
import os
import pickle
from typing import Any

import numpy as np

logger = logging.getLogger("model_inference.mae_mfe")

MODEL_DIR = os.getenv("MODEL_DIR", "/models")
MAE_MODEL_PATH = os.path.join(MODEL_DIR, "mae_predictor.pkl")
MFE_MODEL_PATH = os.path.join(MODEL_DIR, "mfe_predictor.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "mae_mfe_scaler.pkl")

# Batas wajar SL/TP (persen dari harga entry)
MIN_SL_PCT = 0.003   # 0.3% minimum SL
MAX_SL_PCT = 0.08    # 8% maximum SL
MIN_TP_PCT = 0.005   # 0.5% minimum TP
MAX_TP_PCT = 0.20    # 20% maximum TP
SL_SAFETY_MARGIN = 1.15   # Tambah 15% dari prediksi MAE untuk keamanan
TP_CONSERVATIVE = 0.85    # Ambil 85% dari prediksi MFE (lebih aman)

# Encoding regime ke angka
REGIME_ENCODING = {
    "sideways_low_vol": 0.0,
    "trending_up": 1.0,
    "trending_down": 2.0,
    "sideways_high_vol": 3.0,
    "breakout": 4.0,
}


def _build_feature_vector(
    features: dict[str, Any],
    side: str,
    obi: float,
    regime: str,
) -> list[float]:
    """
    Bangun feature vector fixed-length untuk model MAE/MFE.

    Features:
      [0]  atr_pct           : ATR sebagai % dari harga
      [1]  adx_14            : ADX strength
      [2]  rsi_14            : RSI [0, 100]
      [3]  ema_slope_fast    : Slope EMA cepat
      [4]  ema_slope_slow    : Slope EMA lambat
      [5]  volume_ratio      : Volume vs rata-rata
      [6]  bb_width          : Bollinger Band width
      [7]  obi               : Order Book Imbalance [-1, 1]
      [8]  side_encoded      : 1.0 = BUY, -1.0 = SELL
      [9]  regime_encoded    : 0-4 dari REGIME_ENCODING
      [10] bb_position       : Posisi harga di dalam BB [0, 1]
      [11] roc_5             : Rate of Change 5 bar
      [12] ema_cross_9_21   : EMA cross signal
    """
    regime_enc = REGIME_ENCODING.get(regime, 2.0)
    side_enc = 1.0 if str(side).upper() == "BUY" else -1.0

    return [
        float(features.get("atr_pct", 0.015)),
        float(features.get("adx_14", 20.0)),
        float(features.get("rsi_14", 50.0)),
        float(features.get("ema_slope_fast", 0.0)),
        float(features.get("ema_slope_slow", 0.0)),
        float(features.get("volume_ratio", 1.0)),
        float(features.get("bb_width", 0.02)),
        float(obi),
        float(side_enc),
        float(regime_enc),
        float(features.get("bb_position", 0.5)),
        float(features.get("roc_5", 0.0)),
        float(features.get("ema_cross_9_21", 0.0)),
    ]


class MAEMFEPredictor:
    """
    Gradient Boosting Regressor untuk prediksi MAE dan MFE.
    Menghasilkan rekomendasi SL/TP yang dinamis berdasarkan kondisi pasar.
    """

    def __init__(self):
        self._mae_model: Any | None = None
        self._mfe_model: Any | None = None
        self._scaler: Any | None = None
        self._load_models()

    def is_trained(self) -> bool:
        return self._mae_model is not None and self._mfe_model is not None

    def _load_models(self) -> None:
        try:
            if os.path.exists(MAE_MODEL_PATH):
                with open(MAE_MODEL_PATH, "rb") as f:
                    self._mae_model = pickle.load(f)
            if os.path.exists(MFE_MODEL_PATH):
                with open(MFE_MODEL_PATH, "rb") as f:
                    self._mfe_model = pickle.load(f)
            if os.path.exists(SCALER_PATH):
                with open(SCALER_PATH, "rb") as f:
                    self._scaler = pickle.load(f)
            if self.is_trained():
                logger.info("MAE/MFE models loaded from disk")
            else:
                logger.info("MAE/MFE models not found — rule-based fallback will be used")
        except Exception as e:
            logger.warning("Failed to load MAE/MFE models: %s", e)

    def predict(
        self,
        features: dict[str, Any],
        side: str,
        entry_price: float,
        obi: float = 0.0,
        regime: str = "sideways_low_vol",
    ) -> dict[str, Any]:
        """
        Prediksi optimal SL/TP berdasarkan kondisi pasar saat ini.

        Args:
            features: Output dari FeatureEngine.compute_features()
            side: "BUY" atau "SELL"
            entry_price: Harga entry yang direncanakan
            obi: Order Book Imbalance [-1, 1]
            regime: Regime pasar dari GMMRegimeClassifier

        Returns:
            Dict dengan stop_loss, take_profit, mae_pct, mfe_pct, dll.
        """
        if not self.is_trained():
            return self._rule_based_predict(features, side, entry_price, obi, regime)

        try:
            fv = _build_feature_vector(features, side, obi, regime)
            X = np.array([fv])

            if self._scaler:
                X = self._scaler.transform(X)

            mae_pct = float(self._mae_model.predict(X)[0])
            mfe_pct = float(self._mfe_model.predict(X)[0])

            # Clamp ke batas wajar
            mae_pct = max(MIN_SL_PCT, min(mae_pct, MAX_SL_PCT))
            mfe_pct = max(MIN_TP_PCT, min(mfe_pct, MAX_TP_PCT))

            # Hitung harga SL/TP dengan safety margin
            sl_pct = mae_pct * SL_SAFETY_MARGIN
            tp_pct = mfe_pct * TP_CONSERVATIVE

            if str(side).upper() == "BUY":
                stop_loss = entry_price * (1.0 - sl_pct)
                take_profit = entry_price * (1.0 + tp_pct)
            else:
                stop_loss = entry_price * (1.0 + sl_pct)
                take_profit = entry_price * (1.0 - tp_pct)

            rr_ratio = round(tp_pct / sl_pct, 2) if sl_pct > 0 else 0.0

            return {
                "stop_loss": round(stop_loss, 4),
                "take_profit": round(take_profit, 4),
                "mae_pct": round(mae_pct * 100, 3),
                "mfe_pct": round(mfe_pct * 100, 3),
                "sl_pct": round(sl_pct * 100, 3),
                "tp_pct": round(tp_pct * 100, 3),
                "risk_reward_ratio": rr_ratio,
                "source": "ml_model",
                "confidence": 0.80,
                "regime_used": regime,
                "obi_used": round(obi, 4),
            }
        except Exception as e:
            logger.warning("MAE/MFE ML prediction failed: %s — fallback", e)
            return self._rule_based_predict(features, side, entry_price, obi, regime)

    def _rule_based_predict(
        self,
        features: dict[str, Any],
        side: str,
        entry_price: float,
        obi: float,
        regime: str,
    ) -> dict[str, Any]:
        """
        Fallback rule-based ketika model ML belum dilatih.
        Menggunakan ATR + regime + OBI untuk menentukan SL/TP.
        """
        atr_pct = float(features.get("atr_pct", 0.015))

        # Multiplier berdasarkan regime
        regime_multipliers = {
            "trending_up":        (1.8, 3.5),   # (sl_mult, tp_mult)
            "trending_down":      (1.8, 3.5),
            "breakout":           (2.5, 4.0),   # Lebih longgar di breakout
            "sideways_high_vol":  (2.0, 2.0),   # Chop: SL lebar, TP dekat
            "sideways_low_vol":   (1.2, 1.5),   # Konservatif di sideways
        }
        sl_mult, tp_mult = regime_multipliers.get(regime, (1.5, 2.5))

        # OBI boost: sinyal kuat → beri lebih banyak ruang TP
        obi_abs = abs(obi)
        if obi_abs > 0.6:
            tp_mult *= 1.25
        elif obi_abs > 0.4:
            tp_mult *= 1.10

        # OBI kontra sinyal → kurangi exposure
        side_upper = str(side).upper()
        if (side_upper == "BUY" and obi < -0.4) or (side_upper == "SELL" and obi > 0.4):
            sl_mult *= 0.85  # Kurangi SL jika OBI berlawanan

        sl_pct = atr_pct * sl_mult
        tp_pct = atr_pct * tp_mult

        # Clamp
        sl_pct = max(MIN_SL_PCT, min(sl_pct, MAX_SL_PCT))
        tp_pct = max(MIN_TP_PCT, min(tp_pct, MAX_TP_PCT))

        if side_upper == "BUY":
            stop_loss = entry_price * (1.0 - sl_pct)
            take_profit = entry_price * (1.0 + tp_pct)
        else:
            stop_loss = entry_price * (1.0 + sl_pct)
            take_profit = entry_price * (1.0 - tp_pct)

        rr_ratio = round(tp_pct / sl_pct, 2) if sl_pct > 0 else 0.0

        return {
            "stop_loss": round(stop_loss, 4),
            "take_profit": round(take_profit, 4),
            "mae_pct": round(sl_pct * 100 / SL_SAFETY_MARGIN, 3),
            "mfe_pct": round(tp_pct * 100 / TP_CONSERVATIVE, 3),
            "sl_pct": round(sl_pct * 100, 3),
            "tp_pct": round(tp_pct * 100, 3),
            "risk_reward_ratio": rr_ratio,
            "source": "rule_based",
            "confidence": 0.5,
            "regime_used": regime,
            "obi_used": round(obi, 4),
        }

    def train(self, training_data: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Latih model dari data TradeDossier historis.

        training_data: List of dicts dengan keys:
          - features: Dict dari FeatureEngine
          - side: "BUY" | "SELL"
          - obi: float
          - regime: str
          - mae_pct: float (actual MAE dari trade)
          - mfe_pct: float (actual MFE dari trade)
        """
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.preprocessing import StandardScaler

        _min = int(os.getenv("MIN_TRADES_FOR_MAE_MFE", "50"))
        if len(training_data) < _min:
            return {"status": "insufficient_data", "samples": len(training_data), "min_required": _min}

        X, y_mae, y_mfe = [], [], []
        skipped = 0

        for sample in training_data:
            try:
                fv = _build_feature_vector(
                    sample.get("features", {}),
                    sample.get("side", "BUY"),
                    float(sample.get("obi", 0.0)),
                    sample.get("regime", "sideways_low_vol"),
                )
                mae = float(sample["mae_pct"])
                mfe = float(sample["mfe_pct"])

                # Validasi nilai
                if mae < 0 or mfe < 0 or mae > 0.30 or mfe > 0.50:
                    skipped += 1
                    continue

                X.append(fv)
                y_mae.append(mae)
                y_mfe.append(mfe)
            except (KeyError, ValueError, TypeError):
                skipped += 1

        if len(X) < max(10, _min - 5):
            return {"status": "insufficient_valid_data", "valid": len(X), "skipped": skipped}

        X = np.array(X)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        mae_model = GradientBoostingRegressor(
            n_estimators=150, max_depth=5, learning_rate=0.08,
            subsample=0.8, random_state=42,
        )
        mfe_model = GradientBoostingRegressor(
            n_estimators=150, max_depth=5, learning_rate=0.08,
            subsample=0.8, random_state=42,
        )
        mae_model.fit(X_scaled, y_mae)
        mfe_model.fit(X_scaled, y_mfe)

        self._mae_model = mae_model
        self._mfe_model = mfe_model
        self._scaler = scaler

        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(MAE_MODEL_PATH, "wb") as f:
            pickle.dump(mae_model, f)
        with open(MFE_MODEL_PATH, "wb") as f:
            pickle.dump(mfe_model, f)
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)

        logger.info("MAE/MFE models trained. Valid samples: %d, Skipped: %d", len(X), skipped)
        return {
            "status": "trained",
            "valid_samples": len(X),
            "skipped": skipped,
            "mae_importance": mae_model.feature_importances_.tolist(),
            "mfe_importance": mfe_model.feature_importances_.tolist(),
        }
