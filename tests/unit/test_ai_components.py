"""
test_ai_components.py
=======================
Unit tests for the new AI, monitoring, and risk management components:
  1. GMMRegimeClassifier (Market Regime classification & GMM training)
  2. OrderBookMonitor & OBI Calculations (Order book imbalance, liquidity sweeps)
  3. MAEMFEPredictor (Predictive Stop-Loss/Take-Profit, model training)
  4. MacroFilter (Economic news block window for FOMC/CPI)
  5. MLOpsRetrainer (Model training pipeline orchestration)
"""
from __future__ import annotations

import os
import sys
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# Adjust sys.path to import from services
_MODEL_INF_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "services", "model-inference",
)
_RISK_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "services", "risk-gateway",
)
_MARKET_GW_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "services", "market-data-gateway",
)

for d in (_MODEL_INF_DIR, _RISK_DIR, _MARKET_GW_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

from macro_filter import MacroEvent
from mae_mfe_predictor import MAEMFEPredictor
from orderbook_monitor import OrderBookMonitor
from regime_classifier import GMMRegimeClassifier
from retrainer import MLOpsRetrainer

# ─── Mock Data Helpers ─────────────────────────────────────────────────────────

def generate_mock_candles(n_candles: int, trend: float = 0.0) -> list[dict]:
    """Hasilkan mock candles dengan trend (positif = uptrend, negatif = downtrend)."""
    candles = []
    base_price = 50000.0
    now = datetime.now(UTC)

    for i in range(n_candles):
        close_price = base_price + (i * trend) + (np.random.normal(0, 50))
        high_price = close_price + abs(np.random.normal(100, 20))
        low_price = close_price - abs(np.random.normal(100, 20))
        open_price = base_price + ((i - 1) * trend) if i > 0 else close_price

        candles.append({
            "pair": "BTCUSDT",
            "timeframe": "5m",
            "timestamp": (now - timedelta(minutes=5 * (n_candles - i))).isoformat(),
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": abs(np.random.normal(10.0, 2.0)),
        })
    return candles


# ─── 1. GMM Regime Classifier Tests ────────────────────────────────────────────

class TestGMMRegimeClassifier:
    def test_fallback_prediction(self):
        """GMM harus mengembalikan rule-based fallback jika model belum ditrain."""
        classifier = GMMRegimeClassifier(model_path="nonexistent_model.pkl")
        assert not classifier.is_trained()

        candles = generate_mock_candles(10)
        res = classifier.predict(candles)

        assert "regime" in res
        assert "confidence" in res
        assert res["source"] in ("rule_based", "rule_based_fallback")

    def test_gmm_training_and_prediction(self):
        """GMM harus dapat ditrain dari data candle dan memprediksi cluster."""
        classifier = GMMRegimeClassifier(model_path="temp_gmm.pkl")
        candles = generate_mock_candles(120)  # > 100 candle required for GMM

        # Mock sklearn mixture & scale to avoid loading external libraries or writing to actual disk
        with patch("sklearn.mixture.GaussianMixture") as mock_gmm_class, \
             patch("sklearn.preprocessing.StandardScaler") as mock_scaler_class, \
             patch("pickle.dump"):

            mock_gmm = MagicMock()
            mock_gmm.means_ = np.zeros((5, 5))
            mock_gmm.bic.return_value = 100.0
            mock_gmm_class.return_value = mock_gmm

            mock_scaler = MagicMock()
            mock_scaler.fit_transform.return_value = np.zeros((119, 5))
            mock_scaler.transform.return_value = np.zeros((20, 5))
            mock_scaler.inverse_transform.return_value = np.zeros((5, 5))
            mock_scaler_class.return_value = mock_scaler

            res = classifier.train(candles)
            assert res["status"] == "trained"
            assert classifier.is_trained()

            # Predict
            pred = classifier.predict(candles)
            assert "regime" in pred
            assert pred["confidence"] > 0.0


# ─── 2. OrderBookMonitor Tests ─────────────────────────────────────────────────

class TestOrderBookMonitor:
    def test_obi_calculation(self):
        """OBI harus terhitung dengan benar dan memberi sinyal yang pas."""
        monitor = OrderBookMonitor(pairs=["BTCUSDT"], testnet=True)

        # Mock orderbook bids dan asks
        # bids: [[price, volume]]
        bids = [["50000", "30"], ["49990", "30"], ["49980", "40"]]
        asks = [["50010", "5"], ["50020", "5"], ["50030", "10"]]

        # Total bid vol: 100, Total ask vol: 20
        # OBI = (100 - 20) / (100 + 20) = 80 / 120 = 0.667
        snapshot = monitor._calculate_obi("BTCUSDT", bids, asks)

        assert snapshot["obi"] == pytest.approx(0.667, abs=1e-3)
        assert snapshot["signal"] == "BUY"  # OBI > 0.55 (triggers OBI_STRONG_SIGNAL -> BUY directly)
        assert snapshot["liquidity_sweep"] == "none"

    def test_liquidity_sweep_detection(self):
        """OBI harus mendeteksi bid_sweep saat bid volume dominan di 3 level teratas."""
        monitor = OrderBookMonitor(pairs=["BTCUSDT"], testnet=True)

        # Bid volume sangat besar di top levels
        bids = [["50000", "100"], ["49990", "100"], ["49980", "100"]]
        asks = [["50010", "2"], ["50020", "2"], ["50030", "2"]]

        snapshot = monitor._calculate_obi("BTCUSDT", bids, asks)
        assert snapshot["liquidity_sweep"] == "bid_sweep"
        assert snapshot["signal"] == "STRONG_BUY"

        # Ask volume sangat besar di top levels
        bids = [["50000", "2"], ["49990", "2"], ["49980", "2"]]
        asks = [["50010", "100"], ["50020", "100"], ["50030", "100"]]

        snapshot = monitor._calculate_obi("BTCUSDT", bids, asks)
        assert snapshot["liquidity_sweep"] == "ask_sweep"
        assert snapshot["signal"] == "STRONG_SELL"


# ─── 3. MAE/MFE Predictor Tests ────────────────────────────────────────────────

class TestMAEMFEPredictor:
    def test_rule_based_predictions(self):
        """MAE/MFE harus dapat menghasilkan TP/SL logis via rule-based fallback."""
        predictor = MAEMFEPredictor()
        features = {"atr_pct": 0.02}  # ATR 2%

        # BUY side di sideways_low_vol
        res_buy = predictor.predict(
            features=features,
            side="BUY",
            entry_price=100.0,
            obi=0.0,
            regime="sideways_low_vol",
        )
        assert res_buy["stop_loss"] < 100.0
        assert res_buy["take_profit"] > 100.0
        assert res_buy["source"] == "rule_based"
        # TP/SL ratio wajar
        assert res_buy["risk_reward_ratio"] > 0

    def test_mae_mfe_training(self):
        """MAE/MFE harus dapat dilatih menggunakan GradientBoostingRegressor."""
        predictor = MAEMFEPredictor()

        # Generate dummy closed trades data
        training_data = []
        for i in range(60):  # > 50 required
            training_data.append({
                "features": {"atr_pct": 0.015, "rsi_14": 50.0},
                "side": "BUY" if i % 2 == 0 else "SELL",
                "obi": 0.1,
                "regime": "trending_up",
                "mae_pct": 0.015 + np.random.normal(0, 0.002),
                "mfe_pct": 0.035 + np.random.normal(0, 0.005),
            })

        with patch("sklearn.ensemble.GradientBoostingRegressor") as mock_gbr_class, \
             patch("sklearn.preprocessing.StandardScaler") as mock_scaler_class, \
             patch("pickle.dump"):

            mock_model = MagicMock()
            mock_model.feature_importances_ = np.zeros(13)
            mock_gbr_class.return_value = mock_model

            mock_scaler = MagicMock()
            mock_scaler.fit_transform.return_value = np.zeros((60, 13))
            mock_scaler_class.return_value = mock_scaler

            res = predictor.train(training_data)
            assert res["status"] == "trained"
            assert res["valid_samples"] == 60


# ─── 4. Macro Filter Tests ─────────────────────────────────────────────────────

class TestMacroFilter:
    def test_macro_event_window(self):
        """MacroEvent harus memblokir trading di dalam window yang ditentukan."""
        # Event normal (High impact): Block ±30m/60m
        event_time = datetime.now(UTC)
        event = MacroEvent(
            title="Non Farm Payrolls",
            country="USD",
            impact="High",
            dt=event_time,
        )
        assert not event.is_critical

        # 15 menit sebelum event → harus diblokir
        now_before = event_time - timedelta(minutes=15)
        blocked, reason = event.check_blocking(now_before)
        assert blocked
        assert "Non Farm Payrolls" in reason

        # 45 menit setelah event → harus diblokir (cooldown)
        now_after = event_time + timedelta(minutes=45)
        blocked, reason = event.check_blocking(now_after)
        assert blocked

        # 70 menit setelah event → harus lepas blokir
        now_far = event_time + timedelta(minutes=70)
        blocked, _ = event.check_blocking(now_far)
        assert not blocked

    def test_critical_event_multiplier(self):
        """Event CPI/FOMC/Fed (Critical) mendapat block window 2x lebih panjang."""
        event_time = datetime.now(UTC)
        event = MacroEvent(
            title="FOMC Rate Decision",
            country="USD",
            impact="High",
            dt=event_time,
        )
        assert event.is_critical  # Mengandung "fomc"

        # 50 menit sebelum FOMC → harus diblokir (critical block window sebelum = 30 * 2 = 60 menit)
        now_before = event_time - timedelta(minutes=50)
        blocked, _ = event.check_blocking(now_before)
        assert blocked

        # 100 menit setelah FOMC → harus diblokir (critical block window setelah = 60 * 2 = 120 menit)
        now_after = event_time + timedelta(minutes=100)
        blocked, _ = event.check_blocking(now_after)
        assert blocked


# ─── 5. MLOps Retrainer Tests ──────────────────────────────────────────────────

class TestMLOpsRetrainer:
    @pytest.mark.asyncio
    async def test_retrainer_insufficient_data(self):
        """Retrainer harus menolak training jika data di database kurang dari minimum threshold."""
        mock_gmm = MagicMock()
        mock_mae_mfe = MagicMock()
        retrainer = MLOpsRetrainer(mock_gmm, mock_mae_mfe)

        # Mock DB session returns empty results
        with patch("shared.db.session.AsyncSessionLocal") as mock_db:
            mock_session = AsyncMock()
            mock_db.return_value.__aenter__.return_value = mock_session

            mock_result = MagicMock()
            mock_result.scalars.return_value.all.return_value = []
            mock_session.execute = AsyncMock(return_value=mock_result)

            res = await retrainer.run_full_retrain()

            # Harus skip karena tidak ada data
            assert res["gmm"]["status"] == "skipped"
            assert res["mae_mfe"]["status"] == "skipped"
            assert "Insufficient" in res["gmm"]["reason"]
            assert "Insufficient" in res["mae_mfe"]["reason"]
