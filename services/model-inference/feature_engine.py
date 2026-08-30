"""
feature_engine.py
==================
Kalkulasi fitur teknikal dan deteksi market regime untuk model inferensi.
Mendukung: EMA, RSI, ATR, ADX, Bollinger Bands, VWAP, Open Interest ratio.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("model_inference.feature_engine")


class FeatureEngine:
    """Menghitung fitur teknikal dari data candle OHLCV."""

    def __init__(self, feature_version: str = "v1.0"):
        self.feature_version = feature_version

    def compute_features(self, candles: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Hitung semua fitur dari list candle OHLCV.

        Args:
            candles: List dict dengan key open, high, low, close, volume, timestamp
                     diurutkan dari paling lama ke terbaru.

        Returns:
            Dict berisi semua fitur + metadata versi.
        """
        if len(candles) < 50:
            logger.warning("Insufficient candles: %d < 50", len(candles))
            return {}

        closes = np.array([float(c.get("close", 0)) for c in candles])
        highs = np.array([float(c.get("high", 0)) for c in candles])
        lows = np.array([float(c.get("low", 0)) for c in candles])
        volumes = np.array([float(c.get("volume", 0)) for c in candles])

        features: dict[str, Any] = {
            "feature_version": self.feature_version,
            "candle_count": len(candles),
        }

        # EMA crossover
        features["ema_9"] = float(_ema(closes, 9)[-1])
        features["ema_21"] = float(_ema(closes, 21)[-1])
        features["ema_50"] = float(_ema(closes, 50)[-1])
        features["ema_cross_9_21"] = features["ema_9"] - features["ema_21"]
        features["ema_cross_21_50"] = features["ema_21"] - features["ema_50"]

        # RSI
        rsi_arr = _rsi(closes, 14)
        features["rsi_14"] = float(rsi_arr[-1]) if len(rsi_arr) > 0 else 50.0

        # ATR
        atr_arr = _atr(highs, lows, closes, 14)
        features["atr_14"] = float(atr_arr[-1]) if len(atr_arr) > 0 else 0.0
        features["atr_pct"] = (
            features["atr_14"] / closes[-1] * 100.0 if closes[-1] != 0 else 0.0
        )

        # ADX
        adx_val = _adx(highs, lows, closes, 14)
        features["adx_14"] = float(adx_val)

        # Bollinger Bands
        bb_upper, bb_mid, bb_lower = _bollinger(closes, 20, 2.0)
        features["bb_upper"] = float(bb_upper)
        features["bb_mid"] = float(bb_mid)
        features["bb_lower"] = float(bb_lower)
        features["bb_width"] = (
            (bb_upper - bb_lower) / bb_mid if bb_mid != 0 else 0.0
        )
        features["bb_position"] = (
            (closes[-1] - bb_lower) / (bb_upper - bb_lower)
            if (bb_upper - bb_lower) != 0
            else 0.5
        )

        # Volume z-score (20 period)
        if len(volumes) >= 20:
            vol_mean = np.mean(volumes[-20:])
            vol_std = np.std(volumes[-20:]) + 1e-9
            features["volume_zscore"] = float((volumes[-1] - vol_mean) / vol_std)
        else:
            features["volume_zscore"] = 0.0

        # Price position relative to recent range
        high_20 = np.max(highs[-20:]) if len(highs) >= 20 else highs.max()
        low_20 = np.min(lows[-20:]) if len(lows) >= 20 else lows.min()
        features["price_position_20"] = (
            (closes[-1] - low_20) / (high_20 - low_20)
            if (high_20 - low_20) != 0
            else 0.5
        )

        # Rate of change
        features["roc_5"] = _roc(closes, 5)
        features["roc_10"] = _roc(closes, 10)

        # Market regime detection
        features["regime"] = self.detect_regime(features)

        return features

    def detect_regime(self, features: dict[str, Any]) -> str:
        """
        Deteksi market regime berdasarkan ADX, ATR, dan EMA.

        Regimes:
            - trending_bullish: ADX tinggi, EMA bullish
            - trending_bearish: ADX tinggi, EMA bearish
            - sideways_low_vol: ADX rendah, ATR rendah
            - sideways_high_vol: ADX rendah, ATR tinggi
            - breakout: volume spike + BB width expansion
        """
        adx = features.get("adx_14", 20.0)
        ema_cross = features.get("ema_cross_9_21", 0.0)
        atr_pct = features.get("atr_pct", 1.0)
        vol_zscore = features.get("volume_zscore", 0.0)
        bb_width = features.get("bb_width", 0.05)

        if vol_zscore > 2.0 and bb_width > 0.06:
            return "breakout"
        elif adx >= 25:
            if ema_cross > 0:
                return "trending_bullish"
            else:
                return "trending_bearish"
        elif adx < 20:
            if atr_pct > 1.5:
                return "sideways_high_vol"
            else:
                return "sideways_low_vol"
        else:
            return "transitioning"


# -----------------------------------------------------------------------
# Indicator helper functions (pure numpy, no external TA library required)
# -----------------------------------------------------------------------

def _ema(values: np.ndarray, period: int) -> np.ndarray:
    alpha = 2.0 / (period + 1)
    ema = np.zeros_like(values)
    ema[0] = values[0]
    for i in range(1, len(values)):
        ema[i] = alpha * values[i] + (1 - alpha) * ema[i - 1]
    return ema


def _rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    if len(closes) < period + 1:
        return np.array([50.0])
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    rsi_vals = []
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = avg_gain / (avg_loss + 1e-9)
        rsi_vals.append(100.0 - 100.0 / (1.0 + rs))
    return np.array(rsi_vals)


def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    if len(closes) < 2:
        return np.array([0.0])
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1]),
        ),
    )
    atr = np.zeros(len(tr))
    atr[0] = np.mean(tr[: period]) if len(tr) >= period else tr[0]
    alpha = 1.0 / period
    for i in range(1, len(tr)):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]
    return atr


def _adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period * 2:
        return 20.0  # neutral default
    plus_dm = np.zeros(len(highs) - 1)
    minus_dm = np.zeros(len(highs) - 1)
    for i in range(len(plus_dm)):
        up = highs[i + 1] - highs[i]
        down = lows[i] - lows[i + 1]
        plus_dm[i] = up if up > down and up > 0 else 0.0
        minus_dm[i] = down if down > up and down > 0 else 0.0

    atr_arr = _atr(highs, lows, closes, period)
    if len(atr_arr) < period:
        return 20.0

    smooth_plus = np.convolve(plus_dm, np.ones(period) / period, mode="valid")
    smooth_minus = np.convolve(minus_dm, np.ones(period) / period, mode="valid")
    smooth_atr = atr_arr[-len(smooth_plus):]

    plus_di = 100.0 * smooth_plus / (smooth_atr + 1e-9)
    minus_di = 100.0 * smooth_minus / (smooth_atr + 1e-9)
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
    return float(np.mean(dx[-period:])) if len(dx) >= period else 20.0


def _bollinger(closes: np.ndarray, period: int = 20, std_dev: float = 2.0):
    if len(closes) < period:
        c = closes[-1]
        return c, c, c
    window = closes[-period:]
    mid = float(np.mean(window))
    std = float(np.std(window))
    return mid + std_dev * std, mid, mid - std_dev * std


def _roc(closes: np.ndarray, period: int) -> float:
    if len(closes) <= period:
        return 0.0
    prev = closes[-(period + 1)]
    return float((closes[-1] - prev) / (prev + 1e-9) * 100.0)
