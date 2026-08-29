# =============================================================================
# AI TRADING PROFESSIONAL STRATEGY v2.4 (FULLY ENHANCED)
# Freqtrade Futures Strategy - Binance USDT-M Perpetual
# Based on: AI_TRADING_SKILL_v2.md + AI_TRADING_SKILL_v2_ADDENDUM.md
#
# CHANGELOG v2.2: Regime vectorized, confluence MTF, entry simplified, partial TP
# CHANGELOG v2.3: SL trail order fixed, confirm_trade_exit hook, volume fix, RSI short fix
#
# CHANGELOG v2.4 (feature additions):
# [NEW-CRITICAL] Real daily P&L circuit breaker via Trade persistence (bukan per-trade profit)
# [NEW-CRITICAL] Liquidation price check sebelum entry (SL harus sebelum likuidasi)
# [NEW-MAJOR]    BTC/USDT macro filter sebagai @informative (gate altcoin vs BTC arah)
# [NEW-MAJOR]    Freqtrade protections property (CooldownPeriod, StoplossGuard, MaxDrawdown)
# [NEW-MAJOR]    OBV slope + VWAP distance masuk confluence score (Kategori C & D)
# [NEW-MAJOR]    Max trade age timeout exit (24h flat trade = forced exit)
# [NEW-MINOR]    Fee-aware RR: 0.08% round-trip fee diperhitungkan sebelum entry
# [NEW-MINOR]    Real funding rate via exchange.fetch_funding_rate() (bukan ATR proxy)
# [NEW-MINOR]    Correlation guard: tolak entry jika pair terlalu mirip posisi open lain
# [NEW-MINOR]    CustomHyperOptLoss berbasis Calmar Ratio (lebih relevan untuk futures)
# =============================================================================

import logging
import os
import time
import json
from urllib.request import Request, urlopen
from datetime import datetime
from typing import Any, Optional

import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import pandas as pd
import talib.abstract as ta
from freqtrade.optimize.hyperopt import IHyperOptLoss
from freqtrade.persistence import Trade, CustomDataWrapper
from freqtrade.strategy import DecimalParameter, IntParameter, IStrategy, informative
from pandas import DataFrame
from shared.quant.position import position_health, exit_consensus
from shared.quant.allocation import exposure_multiplier
from shared.quant.correlation import pearson, average_correlation

logger = logging.getLogger(__name__)

# Taker fee Binance Futures (VIP0) — dipakai untuk fee-aware RR
EXCHANGE_TAKER_FEE = 0.0004   # 0.04% per side → 0.08% round trip

# ── ML Integration (model-inference service) ─────────────────────────────────
# Helper sync untuk memanggil service ML dari hooks Freqtrade yang synchronous.
# Pakai asyncio.run() + timeout ketat + cache per-pair agar tidak memblokir
# proses setiap candle. Jika service down → fallback fail-open (strategi tetap
# berjalan dengan logika existing), dan kegagalan di-cache sebentar agar tidak
# spam HTTP retry di setiap candle.
MODEL_INFERENCE_URL = os.getenv("MODEL_INFERENCE_URL", "http://model-inference:8000")
_ml_cache: dict[str, dict] = {}          # key: (pair, kind) → {result, ts}
_ml_fail_cache: dict[str, float] = {}    # key: (pair, kind) → epoch ts (backoff)
_ML_CACHE_TTL_S = 55                     # sedikit di bawah 1x candle 5m
_ML_FAIL_BACKOFF_S = 300                 # 5 menit backoff jika service down
# Nonaktifkan ML call di backtest (service ML tidak berjalan, dan HTTP call
# hanya memperlambat). Set true via env untuk force-disable juga.
ML_DISABLED = os.getenv("ML_DISABLED", "false").lower() == "true"
QUANT_ENGINE_URL = os.getenv("QUANT_ENGINE_URL", "http://quant-engine:8000")
QUANT_ENGINE_ENABLED = os.getenv("QUANT_ENGINE_ENABLED", "true").lower() == "true"
QUANT_FACTOR_SCORE_ENABLED = os.getenv("QUANT_FACTOR_SCORE_ENABLED", "false").lower() == "true"
_factor_cache: dict[str, dict] = {}
SENTIMENT_ENGINE_URL = os.getenv("SENTIMENT_ENGINE_URL", "http://sentiment-engine:8000")
SENTIMENT_ENGINE_ENABLED = os.getenv("SENTIMENT_ENGINE_ENABLED", "false").lower() == "true"
_sentiment_cache: dict[str, dict] = {}
NEWS_ALPHA_URL = os.getenv("NEWS_ALPHA_URL", "http://news-alpha:8000")
NEWS_ALPHA_ENABLED = os.getenv("NEWS_ALPHA_ENABLED", "false").lower() == "true"
ON_CHAIN_ENGINE_URL = os.getenv("ON_CHAIN_ENGINE_URL", "http://on-chain-engine:8000")
ON_CHAIN_ENGINE_ENABLED = os.getenv("ON_CHAIN_ENGINE_ENABLED", "false").lower() == "true"
_onchain_cache: dict[str, dict] = {}
POSITION_MONITOR_URL = os.getenv("POSITION_MONITOR_URL", "http://position-monitor:8000")
POSITION_MONITOR_ENABLED = os.getenv("POSITION_MONITOR_ENABLED", "false").lower() == "true"
_position_report_cache: dict[str, float] = {}
ORDERBOOK_INTELLIGENCE_URL = os.getenv("ORDERBOOK_INTELLIGENCE_URL", "http://orderbook-intelligence:8000")
ORDERBOOK_INTELLIGENCE_ENABLED = os.getenv("ORDERBOOK_INTELLIGENCE_ENABLED", "false").lower() == "true"
_orderbook_intelligence_cache: dict[str, dict] = {}


def _quant_params(pair: str, regime: str) -> dict:
    """Fetch adaptive parameters; local strategy values remain the fallback."""
    if not QUANT_ENGINE_ENABLED:
        return {}
    try:
        url = f"{QUANT_ENGINE_URL}/params/{pair.replace('/', '%2F')}?regime={regime}"
        req = Request(url, headers={"Accept": "application/json"})
        with urlopen(req, timeout=0.35) as response:  # noqa: S310
            if response.status == 200:
                payload = json.loads(response.read().decode("utf-8"))
                return payload if isinstance(payload, dict) else {}
    except Exception:  # service outage must not crash Freqtrade hook
        pass
    return {}


def _quant_factor_score(pair: str, side: str, last) -> float | None:
    """Optionally replace the static confluence score with Quant Engine output."""
    if not QUANT_ENGINE_ENABLED or not QUANT_FACTOR_SCORE_ENABLED:
        return None
    key = f"{pair}:{side}"
    cached = _factor_cache.get(key)
    if cached and time.time() - cached.get("ts", 0) < 300:
        return cached.get("score")
    direction = 1.0 if side == "long" else -1.0

    def _number(name: str, default: float = 0.0) -> float:
        try:
            value = float(last.get(name, default) or default)
            return value if np.isfinite(value) else default
        except (TypeError, ValueError):
            return default

    close = max(abs(_number("close", 1.0)), 1e-9)
    ema50 = _number("ema50_4h", _number("ema50", close))
    ema200 = _number("ema200_4h", _number("ema200", close))
    trend = max(-1.0, min(1.0, ((ema50 - ema200) / max(abs(ema200), 1e-9)) * 20.0 * direction))
    momentum = max(-1.0, min(1.0, ((_number("rsi_14", 50.0) - 50.0) / 50.0) * direction))
    mtf = max(-1.0, min(1.0, _number("mtf_alignment", 0.0) * direction))
    volume = max(-1.0, min(1.0, (_number("volume_ratio", 1.0) - 1.0) / 1.5))
    vwap = _number("vwap", close)
    structure = max(-1.0, min(1.0, ((close - vwap) / close) * 20.0 * direction))
    try:
        payload = json.dumps({
            "factors": {"trend": trend, "momentum": momentum, "mtf": mtf,
                        "volume": volume, "structure": structure},
            "weights": {"trend": .25, "momentum": .25, "mtf": .25,
                        "volume": .10, "structure": .15},
        }).encode()
        req = Request(f"{QUANT_ENGINE_URL}/factor-score", data=payload,
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=0.35) as response:  # noqa: S310
            data = json.loads(response.read().decode("utf-8")) if response.status == 200 else {}
        score = float(data.get("score"))
        if np.isfinite(score):
            _factor_cache[key] = {"ts": time.time(), "score": max(0.0, min(100.0, score))}
            return _factor_cache[key]["score"]
    except (TypeError, ValueError, OSError, json.JSONDecodeError):
        pass
    return None


def _news_classify(pair: str, headline: str) -> dict:
    if not NEWS_ALPHA_ENABLED or not headline:
        return {}
    try:
        payload = json.dumps({"pair": pair, "headline": headline}).encode()
        req = Request(f"{NEWS_ALPHA_URL}/classify", data=payload,
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=0.35) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8")) if response.status == 200 else {}
    except Exception:
        return {}


def _sentiment(pair: str) -> dict:
    """Fetch cached Fear & Greed/news sentiment for synchronous strategy hooks."""
    if not SENTIMENT_ENGINE_ENABLED:
        return {}
    cached = _sentiment_cache.get(pair)
    if cached and time.time() - cached.get("ts", 0) < 300:
        return cached.get("data", {})
    try:
        symbol = pair.split("/")[0].split(":")[0]
        req = Request(f"{SENTIMENT_ENGINE_URL}/sentiment/{symbol}",
                      headers={"Accept": "application/json"})
        with urlopen(req, timeout=0.35) as response:  # noqa: S310
            data = json.loads(response.read().decode("utf-8")) if response.status == 200 else {}
        if isinstance(data, dict):
            _sentiment_cache[pair] = {"ts": time.time(), "data": data}
            return data
    except Exception:
        pass
    return {}


def _onchain_metrics(pair: str) -> dict:
    if not ON_CHAIN_ENGINE_ENABLED:
        return {}


def _orderbook_intelligence(pair: str, book: dict) -> dict:
    """Send a normalized DOM snapshot to the optional microstructure service."""
    if not ORDERBOOK_INTELLIGENCE_ENABLED:
        return {}
    try:
        bids = book.get("bids") or []
        asks = book.get("asks") or []
        bid_volume = sum(float(level[1]) for level in bids)
        ask_volume = sum(float(level[1]) for level in asks)
        current_level = max(
            [float(level[1]) for level in bids + asks if len(level) > 1] or [0.0]
        )
        previous = _orderbook_intelligence_cache.get(pair, {})
        payload = json.dumps({
            "bid_volume": bid_volume, "ask_volume": ask_volume,
            "previous_level_size": float(previous.get("level", current_level)),
            "current_level_size": current_level,
            # Trade-flow/refill fields become meaningful when tick-recorder
            # data is supplied by a future caller; zero is an honest default.
            "buy_volume": 0.0, "sell_volume": 0.0,
            "traded_size": 0.0, "refill_count": 0,
            "displayed_size": current_level, "executed_size": 0.0,
        }).encode()
        _orderbook_intelligence_cache[pair] = {"level": current_level, "ts": time.time()}
        req = Request(f"{ORDERBOOK_INTELLIGENCE_URL}/analyze", data=payload,
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=0.25) as response:  # noqa: S310
            result = json.loads(response.read().decode("utf-8")) if response.status == 200 else {}
        return result if isinstance(result, dict) else {}
    except Exception:
        return {}
    cached = _onchain_cache.get(pair)
    if cached and time.time() - cached.get("ts", 0) < 30:
        return cached.get("data", {})
    try:
        req = Request(f"{ON_CHAIN_ENGINE_URL}/metrics/{pair.replace('/', '')}",
                      headers={"Accept": "application/json"})
        with urlopen(req, timeout=0.35) as response:  # noqa: S310
            data = json.loads(response.read().decode("utf-8")) if response.status == 200 else {}
        _onchain_cache[pair] = {"ts": time.time(), "data": data}
        return data
    except Exception:
        return {}


def _record_position_health(pair: str, trade, health: dict, regime: str,
                            current_r: float, peak_r: float, mtf_alignment: float,
                            funding: float = 0.0) -> None:
    """Persist lightweight position telemetry without blocking every candle."""
    if not POSITION_MONITOR_ENABLED or not getattr(trade, "id", None):
        return
    key = str(trade.id)
    now = time.time()
    if now - _position_report_cache.get(key, 0.0) < 60:
        return
    _position_report_cache[key] = now
    try:
        payload = json.dumps({
            "trade_id": key, "pair": pair, "regime": regime,
            "regime_at_entry": getattr(trade, "entry_regime", None) or
                               (getattr(trade, "fb_entry", {}) or {}).get("regime"),
            "current_r": float(current_r), "peak_r": max(0.0, float(peak_r)),
            "mtf_alignment": float(mtf_alignment),
            "notional": float(getattr(trade, "stake_amount", 0) or 0),
            "funding_rate": funding,
            "expected_profit": max(float(current_r), 0.0) * float(getattr(trade, "stake_amount", 0) or 0),
        }).encode()
        req = Request(f"{POSITION_MONITOR_URL}/assess", data=payload,
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=0.25):  # noqa: S310
            pass
    except Exception:
        # Telemetry outage must never alter an exit decision.
        return


def _ml_enabled(self) -> bool:
    """Deteksi apakah ML call sebaiknya aktif (live/dry-run, bukan backtest).

    Freqtrade mengekspos runmode via self.config['runmode']:
      'live' | 'dry_run' → aktif; 'backtest' | 'hyperopt' → nonaktif.
    """
    if ML_DISABLED:
        return False
    try:
        runmode = str(getattr(self, "config", {}).get("runmode", "")).lower()
        if runmode in ("backtest", "hyperopt", "utility"):
            return False
    except Exception:  # noqa: BLE001
        pass
    return True


def _ml_cached(pair: str, kind: str) -> dict | None:
    """Ambil hasil ML dari cache jika masih fresh."""
    item = _ml_cache.get((pair, kind))
    if not item:
        return None
    if time.time() - item["ts"] > _ML_CACHE_TTL_S:
        return None
    return item["result"]


def _ml_call(url_path: str, payload: dict) -> dict | None:
    """Panggil endpoint ML via aiohttp dengan timeout ketat.

    Aman dipanggil dari dalam event loop yang sedang berjalan (live mode
    freqtrade memakai asyncio). Di backtest / saat service down → fail-open.
    """
    import asyncio

    import aiohttp

    async def _do():
        timeout = aiohttp.ClientTimeout(total=3)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{MODEL_INFERENCE_URL}{url_path}", json=payload) as resp:
                if resp.status != 200:
                    return None
                return await resp.json()

    try:
        # Jika sudah ada running loop (live mode), jalankan via ensure_future.
        # Jika tidak (backtest/script), buat loop baru.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_do())
        else:
            return asyncio.get_event_loop().run_until_complete(_do())
    except Exception as e:  # noqa: BLE001
        logger.debug("[ML] call failed %s: %s", url_path, e)
        return None


def _fetch_ml_prediction(pair: str, dataframe: DataFrame, side: str, strategy: Any = None) -> dict | None:
    """Fetch ML probability + regime dari model-inference /predict.

    Cache per-pair. Fail-open: return None jika service down (logika
    strategi existing tetap berjalan). Nonaktif di backtest.
    """
    if strategy is not None and not _ml_enabled(strategy):
        return None

    cache_key = (pair, "predict")
    cached = _ml_cached(pair, "predict")
    if cached is not None:
        return cached

    # Backoff jika baru saja gagal
    if time.time() - _ml_fail_cache.get(cache_key, 0) < _ML_FAIL_BACKOFF_S:
        return None

    try:
        candles = []
        for _, row in dataframe.tail(60).iterrows():
            candles.append({
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "timestamp": pd.Timestamp(row.name).isoformat() if hasattr(row.name, "isoformat") else str(row.name),
            })
        if len(candles) < 50:
            return None

        result = _ml_call("/predict", {
            "pair": pair,
            "timeframe": "5m",
            "candles": candles,
            "strategy_version": "AITradingStrategy",
        })
        if not result:
            _ml_fail_cache[cache_key] = time.time()
            return None

        _ml_cache[cache_key] = {"result": result, "ts": time.time()}
        return result
    except Exception as e:  # noqa: BLE001
        logger.debug("[ML] predict fetch error %s: %s", pair, e)
        _ml_fail_cache[cache_key] = time.time()
        return None


def _fetch_mae_mfe(pair: str, side: str, entry_price: float, dataframe: DataFrame, strategy: Any = None) -> dict | None:
    """Fetch MAE/MFE SL/TP rekomendasi dari model-inference.

    Fail-open: return None jika service down (pakai SL/TP ATR existing).
    Nonaktif di backtest.
    """
    if strategy is not None and not _ml_enabled(strategy):
        return None

    cache_key = (pair, "mae_mfe")
    cached = _ml_cached(pair, "mae_mfe")
    if cached is not None:
        return cached

    if time.time() - _ml_fail_cache.get(cache_key, 0) < _ML_FAIL_BACKOFF_S:
        return None

    try:
        candles = []
        for _, row in dataframe.tail(60).iterrows():
            candles.append({
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "timestamp": pd.Timestamp(row.name).isoformat() if hasattr(row.name, "isoformat") else str(row.name),
            })
        if len(candles) < 50:
            return None

        result = _ml_call("/mae-mfe-predict", {
            "pair": pair,
            "side": "BUY" if side.lower() in ("buy", "long") else "SELL",
            "entry_price": float(entry_price),
            "candles": candles,
        })
        if not result:
            _ml_fail_cache[cache_key] = time.time()
            return None

        _ml_cache[cache_key] = {"result": result, "ts": time.time()}
        return result
    except Exception as e:  # noqa: BLE001
        logger.debug("[ML] mae_mfe fetch error %s: %s", pair, e)
        _ml_fail_cache[cache_key] = time.time()
        return None


# =============================================================================
# [FIX-CRITICAL] MARKET REGIME DETECTOR — VECTORIZED (no lookahead bias)
# Output: pd.Series dengan nilai "TRENDING_BULL"/"TRENDING_BEAR"/"RANGING"/"CHOPPY"
# per baris, dihitung hanya dari data yang ada di atau sebelum baris tersebut.
# =============================================================================
def detect_regime_vectorized(df: DataFrame) -> pd.Series:
    """
    Vectorized regime detection — setiap baris dihitung dari data historis
    sampai baris itu saja. Tidak ada lookahead bias.
    """
    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    ema50  = ta.EMA(close, 50)
    ema200 = ta.EMA(close, 200)
    adx    = ta.ADX(high, low, close, 14)
    atr    = ta.ATR(high, low, close, 14)

    # Rolling range check (choppy)
    rolling_high = high.rolling(20).max()
    rolling_low  = low.rolling(20).min()
    range_pct    = (rolling_high - rolling_low) / (rolling_low + 1e-10) * 100
    avg_atr_pct  = (atr / (close + 1e-10) * 100).rolling(20).mean()

    # Kondisi per baris
    is_trending_bull = (adx > 25) & (close > ema50) & (ema50 > ema200)
    is_trending_bear = (adx > 25) & (close < ema50) & (ema50 < ema200)
    is_choppy        = range_pct < (avg_atr_pct * 3)

    # Priority: trending > choppy > ranging
    regime = pd.Series("RANGING", index=df.index)
    regime = regime.where(~is_choppy,        "CHOPPY")
    regime = regime.where(~is_trending_bear, "TRENDING_BEAR")
    regime = regime.where(~is_trending_bull, "TRENDING_BULL")

    return regime


# =============================================================================
# [FIX-CRITICAL] CONFLUENCE SCORE — VECTORIZED, MTF-AWARE
# Menggunakan kolom _1d / _4h / _1h yang sudah di-inject oleh @informative.
# Output: pd.Series skor 0-100 per baris.
# =============================================================================
def calc_confluence_score(df: DataFrame, direction: str) -> pd.Series:
    """
    Score 0-100 berdasarkan 5 kategori (v2.4: tambah OBV + VWAP + BTC macro):
    A. Trend Proxy / HTF-like  (max 20) — D1 & 4H EMAs + BTC macro alignment
    B. HTF Technical             (max 30) — 4H EMA, structure, volume (4h)
    C. TTF Setup                 (max 25) — BB squeeze, OB/FVG, VWAP distance
    D. ETF Confirmation          (max 25) — candle, BOS, RSI/Stoch/MACD, OBV slope
    """
    s = pd.Series(0.0, index=df.index)
    close = df["close"]

    # ── Kategori A: Trend Proxy + BTC Macro (max 20) ──
    a = pd.Series(0.0, index=df.index)
    if "ema50_1d" in df.columns and "ema200_1d" in df.columns:
        if direction == "long":
            a += np.where(df["ema50_1d"] > df["ema200_1d"], 7, 0)
            a += np.where(close > df["ema50_1d"], 3, 0)
        else:
            a += np.where(df["ema50_1d"] < df["ema200_1d"], 7, 0)
            a += np.where(close < df["ema50_1d"], 3, 0)
    if "adx_1d" in df.columns:
        a += np.where(df["adx_1d"] > 20, 3, 0)
    # [NEW] BTC macro alignment — cegah entry altcoin melawan trend BTC
    if "btc_ema50_1h" in df.columns and "btc_rsi_1h" in df.columns:
        if direction == "long":
            # BTC harus bullish atau netral untuk long altcoin
            a += np.where(df["btc_ema50_1h"] > df["btc_ema50_1h"].shift(3), 4, 0)
            a += np.where(df["btc_rsi_1h"] > 45, 3, 0)
        else:
            # BTC harus bearish atau netral untuk short altcoin
            a += np.where(df["btc_ema50_1h"] < df["btc_ema50_1h"].shift(3), 4, 0)
            a += np.where(df["btc_rsi_1h"] < 55, 3, 0)
    s += a.clip(0, 20)

    # ── Kategori B: HTF Technical (max 30) ──
    b = pd.Series(0.0, index=df.index)
    if "ema50_4h" in df.columns and "ema200_4h" in df.columns:
        if direction == "long":
            b += np.where(close > df["ema50_4h"], 10, 0)
            b += np.where(df["ema50_4h"] > df["ema200_4h"], 5, 0)
        else:
            b += np.where(close < df["ema50_4h"], 10, 0)
            b += np.where(df["ema50_4h"] < df["ema200_4h"], 5, 0)
    if "rsi_4h" in df.columns:
        if direction == "long":
            b += np.where(df["rsi_4h"] > 50, 5, 0)
        else:
            b += np.where(df["rsi_4h"] < 50, 5, 0)
    # Gunakan volume_4h vs volume_ma20_4h (apple-to-apple HTF comparison)
    if "volume_ma20_4h" in df.columns:
        if "volume_4h" in df.columns:
            b += np.where(df["volume_4h"] > df["volume_ma20_4h"], 5, 0)
        elif "volume_ratio" in df.columns:
            b += np.where(df["volume_ratio"] > 1.2, 5, 0)  # Fallback: TTF spike
    if "adx_4h" in df.columns:
        b += np.where(df["adx_4h"] > 25, 5, 0)
    s += b.clip(0, 30)

    # ── Kategori C: TTF Setup + VWAP (max 25) ──
    c = pd.Series(0.0, index=df.index)
    if "bb_squeeze" in df.columns:
        c += np.where(df["bb_squeeze"] == 1, 8, 0)
    if direction == "long":
        if "bos_bull" in df.columns:
            c += np.where(df["bos_bull"] == 1, 6, 0)
        if all(x in df.columns for x in ["high", "low"]):
            fvg_bull = df["low"] > df["high"].shift(2)
            c += np.where(fvg_bull, 5, 0)
        # [NEW] VWAP sebagai dynamic support: harga di atas VWAP = bullish bias
        if "vwap" in df.columns:
            c += np.where(close > df["vwap"], 6, 0)
    else:
        if "bos_bear" in df.columns:
            c += np.where(df["bos_bear"] == 1, 6, 0)
        if all(x in df.columns for x in ["high", "low"]):
            fvg_bear = df["high"] < df["low"].shift(2)
            c += np.where(fvg_bear, 5, 0)
        # [NEW] VWAP sebagai dynamic resistance: harga di bawah VWAP = bearish bias
        if "vwap" in df.columns:
            c += np.where(close < df["vwap"], 6, 0)
    s += c.clip(0, 25)

    # ── Kategori D: ETF Confirmation + OBV Slope (max 25) ──
    d = pd.Series(0.0, index=df.index)
    if "rsi_14" in df.columns:
        if direction == "long":
            d += np.where(df["rsi_14"].between(45, 65), 4, 0)
        else:
            d += np.where(df["rsi_14"].between(55, 75), 4, 0)
    if "stoch_k" in df.columns and "stoch_d" in df.columns:
        if direction == "long":
            d += np.where(df["stoch_k"] > df["stoch_d"], 4, 0)
        else:
            d += np.where(df["stoch_k"] < df["stoch_d"], 4, 0)
    if "macd_hist" in df.columns:
        if direction == "long":
            d += np.where(df["macd_hist"] > 0, 4, 0)
        else:
            d += np.where(df["macd_hist"] < 0, 4, 0)
    if "engulfing" in df.columns:
        if direction == "long":
            d += np.where(df["engulfing"] > 0, 4, 0)
            d += np.where(df.get("hammer", pd.Series(0, index=df.index)) > 0, 2, 0)
        else:
            d += np.where(df["engulfing"] < 0, 4, 0)
            d += np.where(df.get("shooting_star", pd.Series(0, index=df.index)) < 0, 2, 0)
    if "volume_ratio" in df.columns:
        d += np.where(df["volume_ratio"] > 1.5, 2, 0)
    # [NEW] OBV slope: konfirmasi volume momentum searah dengan price
    if "obv_slope" in df.columns:
        if direction == "long":
            d += np.where(df["obv_slope"] > 0, 5, 0)   # OBV naik = akumulasi
        else:
            d += np.where(df["obv_slope"] < 0, 5, 0)   # OBV turun = distribusi
    s += d.clip(0, 25)

    return s.clip(0, 100)


# =============================================================================
# MAIN STRATEGY
# =============================================================================
class AITradingStrategy(IStrategy):
    """
    AI Trading Professional Strategy v2.4
    Freqtrade Futures — Binance USDT-M Perpetual

    v2.4 additions:
    - BTC macro filter (altcoin entry aligned to BTC direction)
    - Real daily P&L circuit breaker (Trade persistence query)
    - Liquidation price check (SL must precede liq price)
    - Freqtrade protections (CooldownPeriod, StoplossGuard, MaxDrawdown)
    - OBV slope + VWAP in confluence score
    - Max trade age timeout (24h flat = exit)
    - Fee-aware RR (0.08% round-trip included)
    - Real funding rate via exchange API
    - Correlation guard (max 2 correlated open positions)
    - CustomHyperOptLoss: Calmar Ratio
    """

    INTERFACE_VERSION = 3
    strategy_name    = "AITradingStrategy"
    timeframe        = "5m"
    can_short        = True

    # ── Risk parameters (aktif dipakai) ──
    # [FIX] Ganti risk_per_trade tiered menjadi satu nilai yang direferensikan
    # custom_stake_amount akan override ini, tapi nilai ini dipakai sebagai baseline
    risk_pct_default = 0.01        # 1% default risk per trade

    # [FIX] min_rrr sekarang dipakai di confirm_trade_entry untuk validasi
    # Skala scalping: RR 1.2+ cukup (TP1 = 1R, runner = 3R)
    # [ENTRY-TIGHTEN] 0.3 → 0.8: entry RR rendah = EV negatif setelah fee.
    # Setup yang lolos harus punya potensi minimal 0.8R per 1R risiko.
    # [ENTRY-TIGHTEN-2] 0.8 → 1.3: mas Yoga mau entry jauh lebih selektif —
    # hanya setup dengan potensi ≥1.3R per 1R risiko yang masuk.
    min_rrr = 1.3                  # Minimum Risk:Reward sebelum entry

    min_confluence_score = 60      # [ENTRY-TIGHTEN] 50 → 60: sinyal lemah di-block
    max_daily_drawdown   = 0.10    # 10% hard stop harian untuk micro-account
    max_consecutive_losses_global = 3  # cooldown entry setelah 3 loss beruntun
    _consecutive_loss_flag = 0
    # ponytail: hard stop 10%; upgrade ke persistent risk service saat multi-instance.
    sl_atr_multiplier    = 1.5     # ATR multiplier untuk SL
    # [RISK-CAP] Batas atas implied SL saat entry. Kalau struktur swing terlalu
    # jauh dari entry (implied SL > cap), setup tidak cocok untuk ukuran akun →
    # tolak di gate, JANGAN buka posisi lalu pasang SL lebar. Sumber risiko
    # (struktur lebar) diblokir sebelum jadi posisi terbuka. 2% price × 5x = 10%.
    max_entry_sl_pct     = 0.02

    # Partial TP levels (R-multiple) — dipakai di adjust_trade_position
    # [FIX 2026-08-28] Turunkan dari 5R/10R/20R → 1.5R/2.5R/4R
    # Data 149 trade: Mean 1.16R, Median 1.00R, Max 5.00R
    # Hanya 0.7% trade capai ≥5R, 7.9% capai ≥2R, 73.6% capai ≥1R
    # TP statis terlalu tinggi = jarang tercapai → stoploss_exit mendominasi
    tp1_rrr       = 1.5     # TP1 di 1.5R (realistis: 73.6% trade capai ≥1R)
    tp2_rrr       = 2.5     # TP2 di 2.5R (realistis: 7.9% trade capai ≥2R)
    tp3_rrr       = 4.0     # TP3 di 4.0R (max observed 5R, buffer 4R cukup)
    tp1_close_pct = 0.40    # Close 40% di TP1
    tp2_close_pct = 0.35    # Close 35% di TP2 (sisanya runner ke TP3)

    # Trailing stop — skala scalping: aktif lebih awal agar profit kecil ter-lock.
    trailing_stop                       = True
    trailing_stop_positive              = 0.004
    trailing_stop_positive_offset       = 0.006
    trailing_only_offset_is_reached     = True

    startup_candle_count: int = 250

    # [ROI-SAFETY-NET] HANYA key "0" statis = single source of truth dgn config
    # generated di services/freqtrade-runtime/main.py ({"0":0.50}). ROI cuma
    # safety-net darurat (profit >50% margin); exit utama = custom_stoploss L1-4
    # trailing + custom_exit. Decay bertahap DIHAPUS: dulu {"0":0.05,...} bikin
    # jual full di +5% sebelum trailing sempat ngejar tren (kasus MUBARAK).
    minimal_roi = {"0": 0.50}
    stoploss    = -0.99             # Dikelola via custom_stoploss
    use_custom_stoploss = True
    process_only_new_candles = False

    order_types = {
        "entry":             "limit",
        "exit":              "limit",
        "stoploss":          "market",
        # Diaktifkan setelah upgrade ke freqtrade 2026.3 + ccxt 4.5.44.
        # Di 2024.3/ccxt 4.2.x, STOP_MARKET gagal dengan error Binance -4120
        # (conditional order dipindah ke endpoint /fapi/v1/algoOrder).
        # 2026.3 memakai endpoint baru + flag {'stop': True} untuk
        # fetch/cancel, sudah diverifikasi manual di container.
        # Tujuan: stoploss dijaga exchange secara real-time, bukan dievaluasi
        # per candle 5m oleh bot (penyebab slippage 2.68% di trade AIO).
        "stoploss_on_exchange": True,
        "stoploss_on_exchange_interval": 60,
    }
    # Trigger stoploss pakai mark price (lebih tahan wick/manipulasi
    # dibanding last price) — konsisten dengan cara Binance hitung likuidasi.
    stoploss_price_type = "mark"

    # ── Hyperopt parameters ──
    adx_threshold        = IntParameter(20, 35, default=25, space="buy", optimize=True)
    volume_spike_factor  = DecimalParameter(1.2, 2.0, default=1.5, space="buy", optimize=True)
    min_conf_long        = IntParameter(40, 85, default=60, space="buy", optimize=True)
    min_conf_short       = IntParameter(40, 85, default=60, space="sell", optimize=True)

    # Max trade hold time (jam) sebelum flat trade di-force exit
    # Skala scalping: trade yang tidak bergerak 3 jam sudah tidak produktif.
    max_trade_age_hours  = 3

    # Korelasi: max pair open dengan "kelompok" BTC/ETH/altcoin besar
    max_correlated_open  = 2

    # ── Internal state ──
    consecutive_losses  : dict = {}
    partial_tp_done     : dict = {}   # Track partial TP per trade_id

    # =========================================================================
    # [NEW] FREQTRADE PROTECTIONS — Safety net bawaan Freqtrade
    # =========================================================================
    @property
    def protections(self):
        return [
            {
                # Cooldown 5 candle setelah setiap trade (kurangi overtrading)
                "method": "CooldownPeriod",
                "stop_duration_candles": 5,
            },
            {
                # Pause pair jika 3 SL dalam 60 candle terakhir
                "method": "StoplossGuard",
                "lookback_period_candles": 60,
                "trade_limit": 3,
                "stop_duration_candles": 60,
                "only_per_pair": True,
            },
            {
                # Global pause jika drawdown > 5% dari 5 trade terakhir
                "method": "MaxDrawdown",
                "lookback_period_candles": 48,
                "trade_limit": 5,
                "max_allowed_drawdown": 0.05,
                "stop_duration_candles": 30,
            },
        ]

    # =========================================================================
    # INFORMATIVE INDICATORS (MTF — @informative decorator)
    # =========================================================================
    @informative("1d")
    def populate_indicators_1d(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Coin baru tanpa data 1d cukup → df kosong → merger throw "empty dataframe".
        # Guard: isi 1 baris NaN → merger OK, semua kolom NaN → no signal (skip pair).
        if dataframe.empty:
            dataframe.loc[0] = [float("nan")] * len(dataframe.columns)
            return dataframe
        dataframe["ema50"]      = ta.EMA(dataframe["close"], 50)
        dataframe["ema200"]     = ta.EMA(dataframe["close"], 200)
        dataframe["rsi"]        = ta.RSI(dataframe["close"], 14)
        dataframe["adx"]        = ta.ADX(dataframe["high"], dataframe["low"], dataframe["close"], 14)
        dataframe["atr"]        = ta.ATR(dataframe["high"], dataframe["low"], dataframe["close"], 14)
        dataframe["volume_ma20"]= dataframe["volume"].rolling(20).mean()
        return dataframe

    @informative("4h")
    def populate_indicators_4h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema50"]       = ta.EMA(dataframe["close"], 50)
        dataframe["ema200"]      = ta.EMA(dataframe["close"], 200)
        dataframe["rsi"]         = ta.RSI(dataframe["close"], 14)
        macd, sig, hist          = ta.MACD(dataframe["close"])
        dataframe["macd_hist"]   = hist
        dataframe["adx"]         = ta.ADX(dataframe["high"], dataframe["low"], dataframe["close"], 14)
        dataframe["atr"]         = ta.ATR(dataframe["high"], dataframe["low"], dataframe["close"], 14)
        dataframe["volume_ma20"] = dataframe["volume"].rolling(20).mean()
        bb_u, bb_m, bb_l         = ta.BBANDS(dataframe["close"], timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe["bb_width"]    = (bb_u - bb_l) / (bb_m + 1e-10)
        return dataframe

    @informative("1h")
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema50"]       = ta.EMA(dataframe["close"], 50)
        dataframe["rsi"]         = ta.RSI(dataframe["close"], 14)
        dataframe["adx"]         = ta.ADX(dataframe["high"], dataframe["low"], dataframe["close"], 14)
        dataframe["atr"]         = ta.ATR(dataframe["high"], dataframe["low"], dataframe["close"], 14)
        dataframe["vwap"]        = qtpylib.rolling_vwap(dataframe)
        dataframe["stoch_k"], dataframe["stoch_d"] = ta.STOCH(
            dataframe["high"], dataframe["low"], dataframe["close"]
        )
        dataframe["volume_ma20"] = dataframe["volume"].rolling(20).mean()
        dataframe["obv"]         = ta.OBV(dataframe["close"], dataframe["volume"])
        return dataframe

    @informative("15m")
    def populate_indicators_15m(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema13"]       = ta.EMA(dataframe["close"], 13)
        dataframe["ema34"]       = ta.EMA(dataframe["close"], 34)
        dataframe["rsi"]         = ta.RSI(dataframe["close"], 14)
        macd, sig, hist          = ta.MACD(dataframe["close"])
        dataframe["macd_hist"]   = hist
        dataframe["atr"]         = ta.ATR(dataframe["high"], dataframe["low"], dataframe["close"], 14)
        dataframe["volume_ma20"] = dataframe["volume"].rolling(20).mean()
        dataframe["adx"]         = ta.ADX(dataframe["high"], dataframe["low"], dataframe["close"], 14)
        return dataframe

    # [NEW] BTC macro filter — inject kolom btc_{column}_1h ke semua pair
    # fmt="btc_{column}_{timeframe}" → ema50 jadi btc_ema50_1h, rsi jadi btc_rsi_1h
    # Hanya BTC/USDT:USDT yang perlu; altcoin lain memakai kolom ini sebagai macro gate.
    @informative("1h", "BTC/USDT:USDT", fmt="btc_{column}_{timeframe}")
    def populate_indicators_btc_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema50"] = ta.EMA(dataframe["close"], 50)
        dataframe["ema200"]= ta.EMA(dataframe["close"], 200)
        dataframe["rsi"]   = ta.RSI(dataframe["close"], 14)
        dataframe["adx"]   = ta.ADX(dataframe["high"], dataframe["low"], dataframe["close"], 14)
        return dataframe

    # =========================================================================
    # POPULATE INDICATORS (5m)
    # =========================================================================
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        close = dataframe["close"]
        high  = dataframe["high"]
        low   = dataframe["low"]
        open_ = dataframe["open"]
        vol   = dataframe["volume"]

        # ── Trend ──
        for p in [8, 13, 21, 34, 50, 89, 144, 200]:
            dataframe[f"ema{p}"] = ta.EMA(close, p)
        for p in [20, 50, 200]:
            dataframe[f"sma{p}"] = ta.SMA(close, p)

        # VWAP
        dataframe["vwap"] = qtpylib.rolling_vwap(dataframe)

        # Ichimoku
        hi9  = high.rolling(9).max();  lo9  = low.rolling(9).min()
        hi26 = high.rolling(26).max(); lo26 = low.rolling(26).min()
        hi52 = high.rolling(52).max(); lo52 = low.rolling(52).min()
        dataframe["ichi_conv"]    = (hi9  + lo9)  / 2
        dataframe["ichi_base"]    = (hi26 + lo26) / 2
        dataframe["ichi_cloud_a"] = (dataframe["ichi_conv"] + dataframe["ichi_base"]) / 2
        dataframe["ichi_cloud_b"] = (hi52 + lo52) / 2

        # SuperTrend (simplified)
        atr7 = ta.ATR(high, low, close, 7)
        hl2  = (high + low) / 2
        dataframe["st_upper"] = hl2 + 3 * atr7
        dataframe["st_lower"] = hl2 - 3 * atr7

        # ── Momentum ──
        dataframe["rsi_14"]   = ta.RSI(close, 14)
        dataframe["rsi_7"]    = ta.RSI(close, 7)
        dataframe["rsi_21"]   = ta.RSI(close, 21)
        dataframe["rsi_slope"]= dataframe["rsi_14"].diff(3)

        macd, sig, hist = ta.MACD(close, 12, 26, 9)
        dataframe["macd"]        = macd
        dataframe["macd_signal"] = sig
        dataframe["macd_hist"]   = hist

        macd_f, sig_f, hist_f = ta.MACD(close, 5, 13, 6)
        dataframe["macd_fast_hist"] = hist_f

        dataframe["stoch_k"], dataframe["stoch_d"] = ta.STOCH(high, low, close, 14, 3, 3)
        dataframe["srsi_k"], dataframe["srsi_d"]   = ta.STOCHRSI(close, 14, 5, 3)
        dataframe["cci"]   = ta.CCI(high, low, close, 14)
        dataframe["willr"] = ta.WILLR(high, low, close, 14)
        dataframe["mfi"]   = ta.MFI(high, low, close, vol, 14)
        dataframe["roc"]   = ta.ROC(close, 10)

        # ── Volatility ──
        dataframe["atr"]      = ta.ATR(high, low, close, 14)
        dataframe["atr_pct"]  = dataframe["atr"] / (close + 1e-10) * 100
        dataframe["atr_avg20"]= dataframe["atr"].rolling(20).mean()
        dataframe["atr_ratio"]= dataframe["atr"] / (dataframe["atr_avg20"] + 1e-10)

        bb_u, bb_m, bb_l = ta.BBANDS(close, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe["bb_upper"] = bb_u
        dataframe["bb_mid"]   = bb_m
        dataframe["bb_lower"] = bb_l
        dataframe["bb_width"] = (bb_u - bb_l) / (bb_m + 1e-10)
        dataframe["bb_pct"]   = (close - bb_l) / (bb_u - bb_l + 1e-10)

        # Keltner Channel
        kc_atr = ta.ATR(high, low, close, 20)
        kc_mid = ta.EMA(close, 20)
        dataframe["kc_upper"] = kc_mid + 1.5 * kc_atr
        dataframe["kc_lower"] = kc_mid - 1.5 * kc_atr

        # BB Squeeze: BB inside KC
        dataframe["bb_squeeze"] = (
            (bb_u < dataframe["kc_upper"]) &
            (bb_l > dataframe["kc_lower"])
        ).astype(int)

        # Donchian
        dataframe["don_high"] = high.rolling(20).max()
        dataframe["don_low"]  = low.rolling(20).min()

        # ── Volume ──
        dataframe["obv"]          = ta.OBV(close, vol)
        dataframe["obv_slope"]    = dataframe["obv"].pct_change(5)
        dataframe["volume_ma20"]  = vol.rolling(20).mean()
        dataframe["volume_ratio"] = vol / (dataframe["volume_ma20"] + 1e-10)
        dataframe["volume_slope"] = vol.pct_change(5)

        mf_mult = ((close - low) - (high - close)) / (high - low + 1e-10)
        dataframe["cmf"] = (mf_mult * vol).rolling(20).sum() / (vol.rolling(20).sum() + 1e-10)

        dataframe["vwma"] = (close * vol).rolling(20).sum() / (vol.rolling(20).sum() + 1e-10)

        # ── Trend Strength ──
        dataframe["adx"]      = ta.ADX(high, low, close, 14)
        dataframe["plus_di"]  = ta.PLUS_DI(high, low, close, 14)
        dataframe["minus_di"] = ta.MINUS_DI(high, low, close, 14)

        # ── Candle Patterns ──
        dataframe["engulfing"]           = ta.CDLENGULFING(open_, high, low, close)
        dataframe["hammer"]              = ta.CDLHAMMER(open_, high, low, close)
        dataframe["shooting_star"]       = ta.CDLSHOOTINGSTAR(open_, high, low, close)
        dataframe["doji"]                = ta.CDLDOJI(open_, high, low, close)
        dataframe["morning_star"]        = ta.CDLMORNINGSTAR(open_, high, low, close)
        dataframe["evening_star"]        = ta.CDLEVENINGSTAR(open_, high, low, close)
        dataframe["three_ws"]            = ta.CDL3WHITESOLDIERS(open_, high, low, close)
        dataframe["three_bc"]            = ta.CDL3BLACKCROWS(open_, high, low, close)
        dataframe["inside_bar"]          = (
            (high < high.shift(1)) & (low > low.shift(1))
        ).astype(int)

        # ── Break of Structure ──
        dataframe["swing_high5"]  = high.rolling(5).max()
        dataframe["swing_low5"]   = low.rolling(5).min()
        dataframe["bos_bull"]     = (close > dataframe["swing_high5"].shift(1)).astype(int)
        dataframe["bos_bear"]     = (close < dataframe["swing_low5"].shift(1)).astype(int)

        # ── Fibonacci (50-candle swing) ──
        rh = high.rolling(50).max()
        rl = low.rolling(50).min()
        fr = rh - rl
        dataframe["fib_236"] = rh - fr * 0.236
        dataframe["fib_382"] = rh - fr * 0.382
        dataframe["fib_500"] = rh - fr * 0.500
        dataframe["fib_618"] = rh - fr * 0.618
        dataframe["fib_786"] = rh - fr * 0.786

        # ── Pivot Points ──
        ph = high.shift(1); pl = low.shift(1); pc = close.shift(1)
        dataframe["pivot"] = (ph + pl + pc) / 3
        dataframe["r1"]    = 2 * dataframe["pivot"] - pl
        dataframe["s1"]    = 2 * dataframe["pivot"] - ph
        dataframe["r2"]    = dataframe["pivot"] + (ph - pl)
        dataframe["s2"]    = dataframe["pivot"] - (ph - pl)

        # ── Candle Analytics ──
        dataframe["body"]              = abs(close - open_)
        dataframe["candle_range"]      = high - low
        dataframe["body_ratio"]        = dataframe["body"] / (dataframe["candle_range"] + 1e-10)
        dataframe["candle_range_ratio"]= dataframe["candle_range"] / (dataframe["atr"] + 1e-10)

        # ── [FIX-CRITICAL] Regime — VECTORIZED, no lookahead ──
        dataframe["regime"] = detect_regime_vectorized(dataframe)

        # Multi-timeframe momentum alignment used by open-position health.
        mtf_parts = []
        if "ema13_15m" in dataframe:
            mtf_parts.append(np.where(dataframe["close"] >= dataframe["ema13_15m"], 1.0, -1.0))
        if "ema50_1h" in dataframe:
            mtf_parts.append(np.where(dataframe["close"] >= dataframe["ema50_1h"], 1.0, -1.0))
        if "ema50_4h" in dataframe:
            mtf_parts.append(np.where(dataframe["close"] >= dataframe["ema50_4h"], 1.0, -1.0))
        dataframe["mtf_alignment"] = np.nanmean(np.vstack(mtf_parts), axis=0) if mtf_parts else 0.0

        # ── Kill Zone (Session) — DISABLED: trading 24/7 ──
        dataframe["session_dead"]   = 0
        dataframe["in_killzone"]    = 1

        # ── Anti-FOMO lock (candle > 2x ATR) ──
        dataframe["big_candle"] = (dataframe["candle_range_ratio"] > 2.0).astype(int)
        dataframe["fomo_lock"]  = dataframe["big_candle"].shift(1).fillna(0).astype(int)

        # ── [FIX-CRITICAL] Confluence Score — pakai kolom MTF informative ──
        # Kolom _1d/_4h/_1h di-inject oleh @informative sebelum populate_indicators dipanggil
        dataframe["conf_score_long"]  = calc_confluence_score(dataframe, "long")
        dataframe["conf_score_short"] = calc_confluence_score(dataframe, "short")

        # Defragment frame: ratusan kolom assignment lewat __setitem__ membuat
        # DataFrame "fragmented" (internal blocks terpisah) -> PerformanceWarning
        # + slowdown per candle. copy() konsolidasi jadi single block.
        return dataframe.copy()

    # =========================================================================
    # [FIX-MAJOR] ENTRY — 3 Hard Gates + Confluence Score
    # (sebelumnya 16 AND conditions yang menyebabkan nyaris 0 trade)
    #
    # STRUKTUR BARU:
    # Gate 1 (HARD): Kill Zone aktif
    # Gate 2 (HARD): Tidak FOMO-lock
    # Gate 3 (HARD): Regime tidak berlawanan arah
    # Gate 4 (SCORE): Confluence >= min_conf
    # Sisa indikator sudah masuk sebagai bobot dalam calc_confluence_score().
    # =========================================================================
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata.get("pair", "")

        # ── ML Confirmation Layer (model-inference) ──
        # Kolom ml_probability/ml_signal_direction diisi dari service ML.
        # Jika service down → NaN → gate ML pass (fail-open), strategi tetap
        # berjalan dengan logika konfluensi existing.
        ml_prob, ml_signal = self._attach_ml_columns(dataframe, pair)
        dataframe["ml_probability"]       = ml_prob
        dataframe["ml_signal_direction"]  = ml_signal

        # A verified, strongly positive headline may justify a breakout that
        # would otherwise be blocked by the anti-FOMO candle lock. This is
        # opt-in and only applies to long entries on the current bar.
        news_override_long = False
        if NEWS_ALPHA_ENABLED and pair and os.getenv("NEWS_HEADLINE", ""):
            news = _news_classify(pair, os.getenv("NEWS_HEADLINE", ""))
            news_override_long = (
                float(news.get("score", 0.0) or 0.0) >= 0.75 and
                news.get("event") == "positive_event"
            )
            if news_override_long and len(dataframe):
                dataframe.loc[dataframe.index[-1], "fomo_lock"] = 0

        # ML gate: hanya aktif jika ML punya data (bukan NaN).
        # Long butuh ML tidak bearish kuat; Short butuh ML tidak bullish kuat.
        ml_long_ok  = dataframe["ml_signal_direction"].isna() | (dataframe["ml_signal_direction"] != "SELL")
        ml_short_ok = dataframe["ml_signal_direction"].isna() | (dataframe["ml_signal_direction"] != "BUY")

        # === LONG ===
        long_gates = (
            (dataframe["in_killzone"]        == 1)         &  # Gate 1: Kill zone
            (dataframe["fomo_lock"]          == 0)         &  # Gate 2: No FOMO
            (dataframe["regime"]             != "TRENDING_BEAR")  &  # Gate 3: Regime
            (dataframe["conf_score_long"]    >= self.min_conf_long.value)  &  # Gate 4: Score
            (dataframe["adx"]                >= self.adx_threshold.value)  &  # Gate 5: Trend
            (dataframe["volume_ratio"]       >= self.volume_spike_factor.value)  &  # Gate 6: Volume
            ml_long_ok                                              # Gate 7: ML confirmation
        )
        dataframe.loc[long_gates, "enter_long"] = 1
        dataframe.loc[long_gates, "enter_tag"]  = "mtf_confluence_long"

        # === SHORT ===
        short_gates = (
            (dataframe["in_killzone"]        == 1)         &
            (dataframe["fomo_lock"]          == 0)         &
            (dataframe["regime"]             != "TRENDING_BULL") &
            (dataframe["conf_score_short"]   >= self.min_conf_short.value) &
            (dataframe["adx"]                >= self.adx_threshold.value) &
            (dataframe["volume_ratio"]       >= self.volume_spike_factor.value) &
            ml_short_ok
        )
        dataframe.loc[short_gates, "enter_short"] = 1
        dataframe.loc[short_gates, "enter_tag"]   = "mtf_confluence_short"

        return dataframe

    def _attach_ml_columns(self, dataframe: DataFrame, pair: str):
        """Ambil probabilitas & signal ML untuk bar terakhir, isi kolom.

        Nilai hanya diisi di baris terakhir (candle yang sedang diproses);
        baris lain diisi NaN agar tidak mempengaruhi backtest historis.
        """
        n = len(dataframe)
        prob = pd.Series(np.nan, index=dataframe.index)
        # object dtype agar bisa menampung string ("BUY"/"SELL"/"HOLD") tanpa
        # FutureWarning "incompatible dtype with float64" (akan jadi error di
        # pandas versi berikutnya).
        signal = pd.Series(np.nan, index=dataframe.index, dtype=object)

        if n < 50 or not pair:
            return prob, signal

        result = _fetch_ml_prediction(pair, dataframe, "both", self)
        if not result:
            return prob, signal

        try:
            p = float(result.get("probability", 0.0))
            sig = str(result.get("signal", "HOLD")).upper()
            if sig in ("BUY", "SELL"):
                prob.iloc[-1]    = p
                signal.iloc[-1]  = sig
            else:
                prob.iloc[-1]    = p
                signal.iloc[-1]  = "HOLD"
        except Exception as e:  # noqa: BLE001
            logger.debug("[ML] attach columns error %s: %s", pair, e)

        return prob, signal

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"]  = 0
        dataframe["exit_short"] = 0
        return dataframe

    # =========================================================================
    # [FIX-MINOR] CUSTOM STOPLOSS — sign diperbaiki & diuji per sisi
    # Freqtrade: return value adalah stoploss relatif ke current_rate.
    # Untuk long: return negatif (misal -0.03 = SL 3% di bawah harga saat ini)
    # Untuk short: return positif (misal +0.03 = SL 3% di atas harga saat ini)
    # =========================================================================
    def _capture_entry_feedback(self, pair: str, trade, dataframe) -> None:
        """Persist entry features once a filled trade has an id."""
        if getattr(trade, "fb_entry", None) or not getattr(trade, "id", None) or len(dataframe) < 1:
            return
        last = dataframe.iloc[-1]
        feature_cols = [
            "ema8", "ema13", "ema21", "ema34", "ema50", "ema89", "ema200",
            "sma20", "sma50", "sma200", "rsi_14", "rsi_7", "rsi_21",
            "macd", "macd_signal", "macd_hist", "stoch_k", "stoch_d",
            "srsi_k", "srsi_d", "cci", "willr", "mfi", "roc", "atr",
            "atr_pct", "atr_ratio", "bb_width", "bb_pct", "bb_squeeze",
            "kc_upper", "kc_lower", "don_high", "don_low", "obv_slope",
            "volume_ratio", "volume_slope", "cmf", "vwma", "adx",
            "ema50_1d", "ema200_1d", "ema50_4h", "ema200_4h", "adx_1d",
            "adx_4h", "rsi_4h", "btc_ema50_1h", "btc_rsi_1h", "mtf_alignment",
        ]
        features = {
            name: float(last[name]) for name in feature_cols
            if name in last.index and isinstance(last[name], (int, float))
            and not pd.isna(last[name])
        }
        features.update({"feature_version": "v1", "candle_count": int(len(dataframe))})
        regime = str(last.get("regime", "RANGING"))
        from shared.feedback import snapshot_entry_conditions
        snapshot_entry_conditions(
            trade=trade,
            regime=regime,
            predicted_rr=float(last.get("predicted_rr", self.tp3_rrr) or self.tp3_rrr),
            ml_signal=str(last.get("ml_signal_direction", "HOLD")),
            ml_prob=float(last.get("ml_probability", 0.5) or 0.5),
            conf=float(last.get("conf_score_short" if trade.is_short else "conf_score_long", 0) or 0),
            atr_ratio=float(last.get("atr_ratio", 1.0) or 1.0),
            side="short" if trade.is_short else "long",
            entry_rate=float(trade.open_rate),
            features=features,
        )
        trade.entry_regime = regime

    def custom_stoploss(self, pair: str, trade, current_time: datetime,
                         current_rate: float, current_profit: float,
                         after_fill: bool, **kwargs) -> float:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) < 1:
            return -0.99

        # [FIX] Persist sl_pct saat fill. `trade.fb_entry` tidak reliable di
        # callback ini (object bisa berbeda), jadi hitung ulang dari candle fill.
        if after_fill and trade.id is not None:
            try:
                self._capture_entry_feedback(pair, trade, dataframe)
                last_fill = dataframe.iloc[-1]
                atr_fill = float(last_fill.get("atr", current_rate * 0.01))
                sl_dist_fill = max(self.sl_atr_multiplier * atr_fill, current_rate * 0.015)
                swing = last_fill.get("swing_high5" if trade.is_short else "swing_low5")
                pad = max(self.sl_atr_multiplier * atr_fill * 0.35, current_rate * 0.0025)
                if swing is not None and not pd.isna(swing):
                    struct_dist = ((float(swing) + pad) - current_rate
                                   if trade.is_short else
                                   current_rate - (float(swing) - pad))
                    sl_dist_fill = max(struct_dist, sl_dist_fill)
                CustomDataWrapper.set_custom_data(
                    trade.id, "sl_pct", float(sl_dist_fill / current_rate))
            except Exception as exc:
                logger.debug(f"[{pair}] sl_pct persist after_fill skipped: {exc}")

        # [LINK-OUTCOME] Persist I(t) orderbook probe ke trade record (biar
        # pas closed nanti, nilai imbalance nempel di trade yang sama — gak
        # perlu grep log manual). Ambil dari _pending_imbalance (di-set di
        # confirm_trade_entry). max_open_trades=1 → aman.
        try:
            if getattr(self, "_pending_imbalance", None):
                CustomDataWrapper.set_custom_data(
                    trade.id, "entry_imbalance",
                    float(self._pending_imbalance["imb"]))
                self._pending_imbalance = None
            if getattr(self, "_pending_orderbook", None) and trade.id is not None:
                CustomDataWrapper.set_custom_data(
                    trade.id, "entry_orderbook_intelligence",
                    dict(self._pending_orderbook))
                self._pending_orderbook = None
        except Exception as exc:
            logger.debug(f"[{pair}] orderbook telemetry persist skipped: {exc}")

        # ── ML MAE/MFE dynamic SL (fail-open) ──
        # Jika model-inference punya rekomendasi SL yang lebih ketat/optimal,
        # gunakan sebagai SL awal (hanya saat posisi masih belum profit).
        # Jika service down / belum profit / data invalid → fallback ATR.
        if current_profit <= 0:
            ml_sl = self._ml_dynamic_stop_loss(pair, trade, dataframe)
            if ml_sl is not None:
                return ml_sl

        last      = dataframe.iloc[-1]
        atr       = last.get("atr", current_rate * 0.01)
        sl_dist   = self.sl_atr_multiplier * atr
        sl_offset = sl_dist * 0.025        # 2.5% anti stop-hunt buffer
        swing_low  = last.get("swing_low5")
        swing_high = last.get("swing_high5")
        # [LAYER-2] Structure-based base stop: bawah/atas swing terakhir.
        # ATR cuma jadi padding, bukan penentu utama.
        struct_pad = max(sl_dist * 0.35, current_rate * 0.0025)
        if swing_low is not None:
            struct_long  = float(swing_low) - struct_pad
        else:
            struct_long  = None
        if swing_high is not None:
            struct_short = float(swing_high) + struct_pad
        else:
            struct_short = None

        open_rate = trade.open_rate
        # 1R = jarak entry→SL yang disimpan saat entry (konsisten dgn custom_exit
        # & adjust_trade_position). ATR live nyusut → R meledak palsu.
        # Sumber kebenaran: CustomDataWrapper (DB native) → fallback fb_entry.
        fb_entry = getattr(trade, "fb_entry", {}) or {}
        sl_pct = None
        try:
            cd = {d.cd_key: d.value for d in
                  CustomDataWrapper.get_custom_data(trade_id=trade.id)}
            sl_pct = cd.get("sl_pct")
        except Exception:
            pass
        if not sl_pct:
            sl_pct = fb_entry.get("sl_pct") or getattr(trade, "sl_pct", None)
        if not sl_pct:
            sl_pct = max((self.sl_atr_multiplier * atr) / open_rate, 0.005)
        # ── [FIX] SL floor 1.5% — kompromi antara proteksi wick & kelancaran entry.
        #    Trade-off: -7.5% margin loss per trade (vs -10% di 2%, -5% di 1%).
        #    ADX filter + trailing L1-4 sudah jadi pengaman exit utama.
        if sl_pct and sl_pct < 0.015:
            logger.debug(f"[{pair}] SL floor override: {sl_pct:.2%} → 1.5%")
            sl_pct = 0.015
        one_r     = float(sl_pct) * open_rate    # 1R dalam harga absolute (konsisten)

        # Hitung current profit dalam harga absolute
        if trade.is_short:
            profit_abs = open_rate - current_rate   # positif jika profit
        else:
            profit_abs = current_rate - open_rate   # positif jika profit

        # [LAYER-3] Grace period: 5 candle pertama (25 menit di TF 5m) setelah
        # entry, SL awal (struktur) doang — tanpa trailing agresif. Entry-noise
        # wajar di candle pertama; trailing baru aktif setelah struktur konfirm.
        try:
            age_candles = (current_time - trade.open_date_utc).total_seconds() / 300.0
        except Exception:
            age_candles = 999.0
        grace_active = age_candles < 5.0

        # [FIX-CRITICAL] Urutan cek dibalik: 3R → 2R → 1R.
        # Progressive lock: makin tinggi profit, SL makin ketat.

        # ── [LAYER-5] Progressive Lock di zona runner (2R+) ──
        # Tambah step lock baru: 3R lock 1.5R, 4R lock 2.5R
        # Ini mencegah runner revert jauh sebelum TP3
        if profit_abs >= 4 * one_r:
            # Lock 2.5R di 4R profit (hampir TP3)
            if trade.is_short:
                sl_price = open_rate - 2.5 * one_r
            else:
                sl_price = open_rate + 2.5 * one_r
        elif profit_abs >= 3 * one_r:
            # Lock 1.5R di 3R profit
            if trade.is_short:
                sl_price = open_rate - 1.5 * one_r
            else:
                sl_price = open_rate + 1.5 * one_r
        elif profit_abs >= 2 * one_r:
            # Lock 0.5R di 2R profit (TP2)
            if trade.is_short:
                sl_price = open_rate - 0.5 * one_r
            else:
                sl_price = open_rate + 0.5 * one_r
        # Trail ke breakeven setelah 1R profit
        elif profit_abs >= one_r:
            sl_price = open_rate       # Breakeven (sama untuk long & short)
        else:
            # SL awal berbasis ATR — tapi dengan FLOOR minimum.
            # ATR 5m di pair murah (DOGE dkk) bisa 0.1%, yang terlalu rapat:
            # kena noise & fee lebih besar dari SL. Floor 0.5% agar SL realistis.
            # [LAYER-2] Kalau struktur tersedia (swing_low5/swing_high5), jarak
            # SL = jarak ke struktur + padding — adaptif, bukan angka arbitrer.
            if trade.is_short and struct_short is not None:
                sl_dist_eff = max((struct_short - open_rate) / 1.0, open_rate * 0.015)
            elif not trade.is_short and struct_long is not None:
                sl_dist_eff = max((open_rate - struct_long) / 1.0, open_rate * 0.015)
            else:
                sl_dist_eff = max(sl_dist, open_rate * 0.015)

            # ── [LAYER-1] Dynamic Trail Percentage ──
            # Trail makin ketat seiring profit naik (0→1R zone):
            # - 0→0.5R: trail 50% peak (longgar, beri ruang)
            # - 0.5R→1R: trail 30% peak (mulai ketat)
            # Ini lebih adaptif daripada fixed 40%
            if trade.is_short:
                sl_initial = open_rate + sl_dist_eff + sl_offset  # SL awal di atas entry
                peak_profit_abs = max(0.0, open_rate - (trade.min_rate or current_rate))

                # Dynamic trail percentage
                if peak_profit_abs >= 0.5 * one_r:
                    trail_pct = 0.30  # 30% trail saat profit 0.5R+
                else:
                    trail_pct = 0.50  # 50% trail saat profit <0.5R

                # Grace period: SL awal doang, tanpa trail — biar entry-noise
                # (1-2 candle) gak langsung sapu. Setelah 5 candle baru trail.
                if grace_active:
                    sl_price = sl_initial
                elif peak_profit_abs > 0 and current_profit >= 0.005:
                    # [LAYER-3] ATR-Based Trail Width
                    # Gunakan ATR live untuk menyesuaikan trail width
                    # Volatility tinggi → trail lebar, volatility rendah → trail sempit
                    atr_trail_width = max(0.5 * atr, current_rate * 0.003)
                    trail_sl = current_rate + (trail_pct * peak_profit_abs)
                    # Pastikan trail tidak lebih lebar dari ATR trail width
                    sl_price = min(sl_initial, max(trail_sl, current_rate + atr_trail_width))
                else:
                    sl_price = sl_initial
                # [SAFETY] SL short HARUS di atas harga minimal 0.15% — cegah -2021.
                sl_price = max(sl_price, current_rate * 1.003)
            else:
                sl_initial = open_rate - sl_dist_eff - sl_offset  # SL awal di bawah entry
                peak_profit_abs = max(0.0, (trade.max_rate or current_rate) - open_rate)

                # Dynamic trail percentage
                if peak_profit_abs >= 0.5 * one_r:
                    trail_pct = 0.30  # 30% trail saat profit 0.5R+
                else:
                    trail_pct = 0.50  # 50% trail saat profit <0.5R

                # Grace period: SL awal doang, tanpa trail — biar entry-noise
                # (1-2 candle) gak langsung sapu. Setelah 5 candle baru trail.
                if grace_active:
                    sl_price = sl_initial
                elif peak_profit_abs > 0 and current_profit >= 0.005:
                    # [LAYER-3] ATR-Based Trail Width
                    atr_trail_width = max(0.5 * atr, current_rate * 0.003)
                    trail_sl = current_rate - (trail_pct * peak_profit_abs)
                    sl_price = max(sl_initial, min(trail_sl, current_rate - atr_trail_width))
                else:
                    sl_price = sl_initial
                # [SAFETY] SL long HARUS di bawah harga minimal 0.15% — cegah -2021.
                sl_price = min(sl_price, current_rate * 0.997)

            # ── [LAYER-4] Ratchet SL — SL hanya boleh NAIK, tidak boleh TURUN ──
            # Ambil SL sebelumnya dari CustomDataWrapper
            # Ini memastikan SL tidak pernah turun (hanya naik atau tetap)
            try:
                prev_sl_data = CustomDataWrapper.get_custom_data(
                    trade_id=trade.id, cd_key="current_sl"
                )
                if prev_sl_data:
                    prev_sl = float(prev_sl_data[0].value)
                    # Ratchet: hanya naik, tidak pernah turun
                    if trade.is_short:
                        sl_price = min(sl_price, prev_sl)  # Short: SL di atas, makin kecil makin baik
                    else:
                        sl_price = max(sl_price, prev_sl)  # Long: SL di bawah, makin besar makin baik
            except Exception:
                pass  # Fail-open, pakai SL yang dihitung

        # ── [LAYER-6] Simpan SL saat ini ke CustomDataWrapper ──
        # Untuk ratchet di call berikutnya
        try:
            CustomDataWrapper.set_custom_data(
                trade.id, "current_sl", float(sl_price)
            )
        except Exception:
            pass

        # [LAYER-4] Invalidation by CLOSE, bukan wick: SL dievaluasi terhadap
        # close candle terakhir (bukan current_rate live) — wick nembus level
        # tapi candle close balik = noise, bukan sinyal reversal. Harga yang
        # dipakai buat hitung jarak SL = close candle, bukan tick live.
        try:
            last_close = float(dataframe.iloc[-1]["close"])
        except Exception:
            last_close = current_rate
        ref_rate = last_close if last_close > 0 else current_rate

        # [FIX] Konversi ke relatif terhadap ref_rate dengan tanda yang benar
        if trade.is_short:
            # Untuk short: SL di atas ref_rate → return positif
            stoploss_pct = (sl_price - ref_rate) / ref_rate
            # Pastikan tidak negatif (SL short harus di atas harga)
            stoploss_pct = max(0.003, stoploss_pct)
        else:
            # Untuk long: SL di bawah ref_rate → return negatif
            stoploss_pct = (sl_price - ref_rate) / ref_rate
            # Pastikan tidak positif (SL long harus di bawah harga)
            stoploss_pct = min(-0.003, stoploss_pct)

        # [FIX-TRIGGER] Cegah "Order would immediately trigger" (-2021).
        # Kalau SL price sudah tidak valid (SL long > current price, atau
        # SL short < current price), skip update — biarkan SL lama tetap aktif.
        # Update hanya kalau SL price masih di area yang valid.
        if trade.is_short:
            # SL short harus di ATAS current price
            if sl_price <= current_rate:
                logger.debug(
                    f"[{pair}] SL short {sl_price:.4f} <= current {current_rate:.4f} — skip update"
                )
                return None  # Skip update, biarkan SL lama
        else:
            # SL long harus di BAWAH current price
            if sl_price >= current_rate:
                logger.debug(
                    f"[{pair}] SL long {sl_price:.4f} >= current {current_rate:.4f} — skip update"
                )
                return None  # Skip update, biarkan SL lama

        # [FIX-EMERGENCY] Cegah "Order would immediately trigger" (-2021) setelah restart.
        # Kalau harga pasar SUDAH melewati stop baru (mis. gap harga saat bot down/restart),
        # freqtrade bakal gagal pasang SL → panic emergency_exit di loss lebih dalam.
        # Solusi: kembalikan SL yang lebih longgar (kearah entry) biar bisa dipasang.
        # [LAYER-4] Pakai ref_rate (close candle) sebagai pembanding, bukan
        # current_rate live — wick sesaat yang lewat SL gak dianggap invalidasi.
        if trade.is_short:
            if ref_rate >= sl_price:  # close sudah di atas SL short → SL tak terpasang
                loosened = open_rate + max(sl_dist_eff, sl_dist, one_r) * 1.5
                logger.warning(
                    f"[EMERGENCY-LOOSEN-ALERT] [{pair}] SL short {sl_price:.4f} sudah terlewati close {ref_rate:.4f} "
                    f"— longgarkan SL ke {loosened:.4f} (risk > cap normal, darurat restart/gap). "
                    f"AKSI: cek posisi, pertimbangkan close manual atau terima risiko sekali ini."
                )
                stoploss_pct = max((loosened - ref_rate) / ref_rate, 0.001)
        else:
            if ref_rate <= sl_price:  # close sudah di bawah SL long → SL tak terpasang
                loosened = open_rate - max(sl_dist_eff, sl_dist, one_r) * 1.5
                logger.warning(
                    f"[EMERGENCY-LOOSEN-ALERT] [{pair}] SL long {sl_price:.4f} sudah terlewati close {ref_rate:.4f} "
                    f"— longgarkan SL ke {loosened:.4f} (risk > cap normal, darurat restart/gap). "
                    f"AKSI: cek posisi, pertimbangkan close manual atau terima risiko sekali ini."
                )
                stoploss_pct = min((loosened - ref_rate) / ref_rate, -0.001)

        return stoploss_pct

    def _ml_dynamic_stop_loss(self, pair: str, trade, dataframe: DataFrame) -> float | None:
        """SL dinamis dari MAE/MFE model (fail-open).

        Return relative stoploss yang siap dikembalikan custom_stoploss,
        atau None jika ML tidak tersedia / rekomendasi invalid.
        Juga menyimpan TP dinamis rekomendasi ML ke state trade agar
        custom_exit bisa memakainya (dynamic take-profit).
        """
        try:
            entry_rate = float(trade.open_rate)
            if entry_rate <= 0:
                return None
            side = "short" if trade.is_short else "long"
            ml = _fetch_mae_mfe(pair, side, entry_rate, dataframe, self)
            if not ml:
                return None

            sl_price = ml.get("stop_loss")
            if sl_price is None:
                return None
            sl_price = float(sl_price)
            if sl_price <= 0:
                return None

            # ── Simpan TP dinamis ke state trade (dipakai custom_exit) ──
            # ML MAE/MFE menghitung take_profit optimal dari data historis —
            # ini membuat TP adaptif, bukan R-multiple statis.
            tp_price = ml.get("take_profit")
            if tp_price:
                try:
                    tp_f = float(tp_price)
                    if tp_f > 0:
                        if not hasattr(trade, "ml_sl_tp"):
                            trade.ml_sl_tp = {}
                        trade.ml_sl_tp["take_profit"] = tp_f
                        trade.ml_sl_tp["sl_price"] = sl_price
                        trade.ml_sl_tp["side"] = side
                        trade.ml_sl_tp["entry"] = entry_rate
                except (TypeError, ValueError):
                    pass

            # Validasi arah: long → SL harus di bawah entry; short → di atas entry
            if trade.is_short and sl_price <= entry_rate:
                return None
            if not trade.is_short and sl_price >= entry_rate:
                return None

            # Floor SL minimum 0.5% — sama dengan custom_stoploss.
            # ML bisa rekomendasi SL terlalu rapat di ATR 5m kecil.
            min_sl_dist = entry_rate * 0.005
            if abs(sl_price - entry_rate) < min_sl_dist:
                return None

            # Pakai harga terakhir dataframe sebagai acuan aktual
            current_rate = float(dataframe["close"].iloc[-1]) if len(dataframe) > 0 else float(trade.open_rate)
            if current_rate <= 0:
                return None

            pct = (sl_price - current_rate) / current_rate
            # Long → negatif (SL di bawah harga); Short → positif (SL di atas harga)
            return max(0.003, pct) if trade.is_short else min(-0.003, pct)
        except Exception as e:  # noqa: BLE001
            logger.debug("[ML] dynamic SL error %s: %s", pair, e)
            return None

    # =========================================================================
    # [FIX-MAJOR] PARTIAL TP via adjust_trade_position()
    # TP1 (1.5R) → close 30%, TP2 (2.5R) → close 40%, runner ke TP3 (4R)
    # =========================================================================
    def adjust_trade_position(self, trade, current_time: datetime,
                               current_rate: float, current_profit: float,
                               min_stake: float | None, max_stake: float,
                               current_entry_rate: float, current_exit_rate: float,
                               current_entry_profit: float, current_exit_profit: float,
                               **kwargs) -> float | None:
        """
        Partial close di TP1 dan TP2.
        Return nilai negatif = reduce position (close sebagian).
        Return None = tidak ada perubahan.
        """
        if trade.id not in self.partial_tp_done:
            self.partial_tp_done[trade.id] = {"tp1": False, "tp2": False}
            # Recover state dari CustomDataWrapper (persist lintas restart di DB)
            try:
                cd = {d.cd_key: d.value for d in
                      CustomDataWrapper.get_custom_data(trade_id=trade.id)}
                if cd.get("tp1_done"):
                    self.partial_tp_done[trade.id]["tp1"] = True
                if cd.get("tp2_done"):
                    self.partial_tp_done[trade.id]["tp2"] = True
            except Exception:
                pass

        state = self.partial_tp_done[trade.id]

        # Hitung R-multiple saat ini
        pair      = trade.pair
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) < 1:
            return None

        atr    = dataframe["atr"].iloc[-1]
        # 1R dari jarak SL yang disimpan saat entry (bukan ATR live yg nyusut).
        # Sumber kebenaran: CustomDataWrapper (DB native) → fallback fb_entry.
        fb_entry = getattr(trade, "fb_entry", {}) or {}
        sl_pct = None
        try:
            cd = {d.cd_key: d.value for d in
                  CustomDataWrapper.get_custom_data(trade_id=trade.id)}
            sl_pct = cd.get("sl_pct")
        except Exception:
            pass
        if not sl_pct:
            sl_pct = fb_entry.get("sl_pct") or getattr(trade, "sl_pct", None)
        if not sl_pct:
            sl_pct = max((self.sl_atr_multiplier * atr) / trade.open_rate, 0.005)
        sl_pct = float(sl_pct)
        one_r  = sl_pct  # 1R dalam fraksi (mis. 0.0157 = 1.57%)

        if one_r <= 0:
            return None

        r_multiple = current_profit / one_r
        regime = str(dataframe.iloc[-1].get("regime", "unknown"))
        adaptive = _quant_params(pair, regime)
        tp1_rrr = float(adaptive.get("tp1_rrr", self.tp1_rrr))
        tp2_rrr = float(adaptive.get("tp2_rrr", self.tp2_rrr))
        atr_ratio = float(dataframe.iloc[-1].get("atr_ratio", 1.0) or 1.0)
        tp1_close_pct = self.tp1_close_pct
        tp2_close_pct = self.tp2_close_pct
        if atr_ratio > 2.0:
            # Volatility-adjusted partials: realize more earlier when the
            # chance of a runner reverting is materially higher.
            tp1_close_pct = min(0.50, tp1_close_pct + 0.10)
            tp2_close_pct = min(0.45, tp2_close_pct + 0.05)

        # TP1: quant-engine calibrated R level → close adaptive percentage
        if not state["tp1"] and r_multiple >= tp1_rrr:
            state["tp1"] = True
            close_stake = trade.stake_amount * tp1_close_pct
            logger.info(f"[{pair}] TP1 hit ({r_multiple:.2f}R) → close {tp1_close_pct:.0%}")
            # Persist state biar gak reset kalau restart bot (DB native)
            try:
                CustomDataWrapper.set_custom_data(trade.id, "tp1_done", True)
            except Exception:
                pass
            # Return stake-currency amount (negatif = reduce position)
            return -close_stake


        # TP2: quant-engine calibrated R level → close adaptive percentage
        if not state["tp2"] and r_multiple >= tp2_rrr:
            state["tp2"] = True
            close_stake = trade.stake_amount * tp2_close_pct
            logger.info(f"[{pair}] TP2 hit ({r_multiple:.2f}R) → close {tp2_close_pct:.0%}")
            try:
                CustomDataWrapper.set_custom_data(trade.id, "tp2_done", True)
            except Exception:
                pass
            # Return stake-currency amount (negatif = reduce position)
            return -close_stake

        # Runner (sisa 30%) berjalan ke TP3 via trailing SL
        return None

    # =========================================================================
    # CUSTOM EXIT
    # =========================================================================
    def custom_exit(self, pair: str, trade, current_time: datetime,
                     current_rate: float, current_profit: float,
                     **kwargs) -> str | None:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) < 1:
            return None

        last   = dataframe.iloc[-1]
        atr    = last.get("atr", current_rate * 0.01)
        # 1R = jarak entry→SL yang disimpan saat entry (akurat, ATR-live nyusut).
        # Sumber kebenaran: CustomDataWrapper (DB native) → fallback fb_entry.
        fb_entry = getattr(trade, "fb_entry", {}) or {}
        sl_pct = None
        try:
            cd = {d.cd_key: d.value for d in
                  CustomDataWrapper.get_custom_data(trade_id=trade.id)}
            sl_pct = cd.get("sl_pct")
        except Exception:
            pass
        if not sl_pct:
            sl_pct = fb_entry.get("sl_pct") or getattr(trade, "sl_pct", None)
        if not sl_pct:
            sl_pct = (self.sl_atr_multiplier * atr) / trade.open_rate
            sl_pct = max(float(sl_pct), 0.005)
        sl_pct = float(sl_pct)
        one_r  = sl_pct
        r_mult = current_profit / one_r if one_r > 0 else 0
        consensus_enabled = os.getenv("EXIT_CONSENSUS_ENABLED", "false").lower() == "true"
        health = {"score": 100.0, "momentum_decay": 0.0, "thesis_valid": True}
        peak_r_mult = max(0.0, r_mult)
        mtf_alignment = float(last.get("mtf_alignment", 1.0) or 1.0)
        # Trade-health telemetry is always safe; exit consensus is opt-in until
        # enough historical calibration exists.
        try:
            peak_rate = float(getattr(trade, "min_rate", 0) or 0) if trade.is_short else float(getattr(trade, "max_rate", 0) or 0)
            peak_profit = ((float(trade.open_rate) - peak_rate) / float(trade.open_rate)
                           if trade.is_short else
                           (peak_rate - float(trade.open_rate)) / float(trade.open_rate))
            peak_r_mult = max(0.0, peak_profit / one_r) if one_r > 0 else r_mult
            health = position_health(
                r_mult, peak_r_mult, str(last.get("regime", "unknown")),
                getattr(trade, "entry_regime", None) or (fb_entry.get("regime") if fb_entry else None),
                mtf_alignment,
            )
            logger.debug("[%s] trade_health=%s", pair, health)
        except Exception as e:
            logger.debug("[%s] trade health skipped: %s", pair, e)

        # ── [NEW] Real daily P&L circuit breaker ──
        # Hitung total P&L USDT dari semua trade yang ditutup hari ini
        try:
            today_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
            closed_today = Trade.get_trades([
                Trade.close_date >= today_start,
                Trade.is_open.is_(False),
            ]).all()
            daily_pnl = sum(
                t.close_profit_abs for t in closed_today
                if t.close_profit_abs is not None
            )
            try:
                total_capital = self.wallets.get_total("USDT")
            except Exception:
                total_capital = 1000.0
            daily_loss_pct = daily_pnl / (total_capital + 1e-10)
            if daily_loss_pct < -self.max_daily_drawdown:
                logger.warning(
                    f"[{pair}] DAILY DRAWDOWN BREAKER: {daily_loss_pct:.2%} loss today"
                )
                return "daily_drawdown_circuit_breaker"
        except Exception as e:
            logger.debug(f"[{pair}] Daily P&L check skipped: {e}")

        # ── [HARDEN] GLOBAL consecutive-loss breaker (portfolio-wide) ──
        # Cegah spiral negatif: kalau sudah rugi N trade beruntun hari ini,
        # stop buka posisi baru (block entry lewat confirm_trade_entry).
        try:
            today_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
            closed_today = Trade.get_trades([
                Trade.close_date >= today_start,
                Trade.is_open.is_(False),
            ]).all()
            recent = [t for t in closed_today
                      if t.close_profit_abs is not None and t.close_profit_abs < 0]
            # Hitung beruntun dari trade terakhir mundur
            consecutive = 0
            for t in sorted(closed_today, key=lambda x: x.close_date or x.close_date_utc, reverse=True):
                if t.close_profit_abs is not None and t.close_profit_abs < 0:
                    consecutive += 1
                else:
                    break
            if consecutive >= self.max_consecutive_losses_global:
                logger.warning(
                    f"[{pair}] GLOBAL CONSECUTIVE LOSS BREAKER: {consecutive} rugi beruntun"
                )
                # Set flag di persistor supaya confirm_trade_entry blokir entry baru
                try:
                    self._consecutive_loss_flag = consecutive
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"[{pair}] Consecutive-loss check skipped: {e}")

        # ── [NEW] Max trade age timeout (flat trade) ──
        hours_open = (current_time - trade.open_date_utc).total_seconds() / 3600
        if hours_open > self.max_trade_age_hours and abs(current_profit) < 0.003:
            logger.info(f"[{pair}] TIMEOUT: flat trade {hours_open:.1f}h, profit {current_profit:.3%}")
            return "timeout_flat_exit"

        # ── Regime change exit ──
        regime = last.get("regime", "RANGING")
        adaptive_exit = _quant_params(pair, str(regime))
        tp3_rrr = float(adaptive_exit.get("tp3_rrr", self.tp3_rrr))
        regime_exit = (not trade.is_short and regime == "TRENDING_BEAR") or (trade.is_short and regime == "TRENDING_BULL")
        if regime_exit and not consensus_enabled:
            return "regime_change_to_bear" if not trade.is_short else "regime_change_to_bull"
        funding_exit = False
        funding = 0.0

        # Funding cost guard while a position is open. Only charge funding in
        # the adverse direction for the position side.
        if os.getenv("POSITION_RISK_ENABLED", "false").lower() == "true":
            try:
                funding = float(last.get("funding_rate", 0.0) or 0.0)
                adverse = (not trade.is_short and funding > 0) or (trade.is_short and funding < 0)
                periods = max(1, int(hours_open / 8))
                notional = float(getattr(trade, "stake_amount", 0)) * float(getattr(trade, "leverage", 1) or 1)
                funding_cost = notional * abs(funding) * periods
                expected_profit = abs(float(getattr(trade, "stake_amount", 0))) * max(current_profit, 0)
                if adverse and expected_profit > 0 and funding_cost >= expected_profit * .5:
                    funding_exit = True
            except Exception as e:
                logger.debug("[%s] funding impact skipped: %s", pair, e)

        _record_position_health(
            pair, trade, health, str(regime), r_mult, peak_r_mult,
            mtf_alignment, funding,
        )

        # ── Reversal kuat (ADX + volume + dual-candle confirmation) ──
        # [UPDATE 2026-08-26] Filter ganda anti-false-signal di 5m:
        #  (A) ADX < 30 → tren lemah/ranging (saat kuat pattern = pullback)
        #  (B) Volume ratio >= 1.3x → konfirmasi institusi, bukan retail noise
        #  (C) Dual-candle confirmation: 2 candle berturut-turut signal reversal
        # Pattern reversal 5m tanpa filter ini false-positive rate tinggi.
        # Lolos ketiganya = reversal kuat, boleh full-exit.
        adx_now = last.get("adx", 25)
        volume_now = float(last.get("volume_ratio", 1.0) or 1.0)
        prev = dataframe.iloc[-2] if len(dataframe) >= 2 else last
        reversal_signal = False

        if current_profit > 0 and float(adx_now or 0) < 30 and volume_now >= 1.3:
            # Konfirmasi 2 candle: pattern sekarang + 1 candle sebelumnya searah
            if not trade.is_short:
                cur_signal = (last.get("engulfing", 0) < 0 or last.get("shooting_star", 0) < 0)
                prev_signal = (prev.get("engulfing", 0) < 0 or prev.get("shooting_star", 0) < 0)
                if cur_signal and prev_signal:
                    reversal_signal = True
            else:
                cur_signal = (last.get("engulfing", 0) > 0 or last.get("hammer", 0) > 0)
                prev_signal = (prev.get("engulfing", 0) > 0 or prev.get("hammer", 0) > 0)
                if cur_signal and prev_signal:
                    reversal_signal = True

        if consensus_enabled:
            signals = {
                "regime": regime_exit,
                "ml": bool(getattr(trade, "ml_exit_signal", False)),
                "momentum": health.get("momentum_decay", 0.0) > 0.35,
                "reversal": reversal_signal,
                "volume": volume_now < 0.7,
                "funding": funding_exit,
            }
            should_exit, score = exit_consensus(
                signals, threshold=float(os.getenv("EXIT_CONSENSUS_THRESHOLD", ".65"))
            )
            if should_exit:
                logger.info("[%s] Exit consensus score %.2f signals=%s", pair, score, signals)
                return "exit_consensus"
        elif funding_exit:
            return "funding_cost_exceed_exit"

        # ── Dynamic TP dari ML MAE/MFE ──
        # Jika model merekomendasikan take_profit spesifik (dari data historis),
        # exit penuh saat harga mencapai level itu. Lebih pintar dari R statis
        # karena menyesuaikan kondisi pasar saat entry.
        ml_tp = getattr(trade, "ml_sl_tp", None)
        if ml_tp and ml_tp.get("take_profit"):
            try:
                tp_price = float(ml_tp["take_profit"])
                if tp_price > 0:
                    if (not trade.is_short and current_rate >= tp_price) or \
                       (trade.is_short and current_rate <= tp_price):
                        logger.info(
                            f"[{pair}] Dynamic TP hit: {current_rate:.4f} "
                            f"(ML TP {tp_price:.4f})"
                        )
                        return "ml_dynamic_tp"
            except (TypeError, ValueError):
                pass

        # ── TP3 / Full runner exit ──
        if r_mult >= tp3_rrr:
            return "tp3_full_runner_exit"

        # ── Dead zone exit dengan profit kecil ──
        # Skala scalping: profit 0.3%+ sudah layak di-lock saat session mati.
        if last.get("session_dead", 0) == 1 and current_profit > 0.003:
            return "dead_session_profit_lock"

        return None

    # =========================================================================
    # ADAPTIVE POSITION SIZING
    # =========================================================================
    def custom_stake_amount(self, current_time: datetime, current_rate: float,
                             proposed_stake: float, min_stake: float | None,
                             max_stake: float, entry_tag: str | None,
                             side: str, **kwargs) -> float:
        try:
            total_capital = self.wallets.get_total("USDT")
        except Exception:
            total_capital = proposed_stake * 10

        # Tier risiko berdasarkan ukuran modal
        if total_capital < 10:
            risk_pct = 0.005
        elif total_capital < 100:
            risk_pct = 0.008
        elif total_capital < 1000:
            risk_pct = 0.010
        else:
            risk_pct = 0.010

        if os.getenv("CAPITAL_ALLOCATION_ENABLED", "false").lower() == "true":
            try:
                pair_for_alloc = kwargs.get("pair") or ""
                frame, _ = self.dp.get_analyzed_dataframe(pair_for_alloc, self.timeframe)
                regime = str(frame.iloc[-1].get("regime", "unknown")) if len(frame) else "unknown"
                closed = Trade.get_trades([Trade.is_open.is_(False)]).all()
                pnl = sum(float(t.close_profit_abs or 0) for t in closed)
                drawdown = max(0.0, -pnl / max(float(total_capital), 1e-9))
                risk_pct *= exposure_multiplier(drawdown, regime)
            except Exception as e:
                logger.debug("Capital allocation skipped: %s", e)

        risk_amount = total_capital * risk_pct

        pair = kwargs.get("pair", "BTCUSDT")
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if len(dataframe) > 0:
            last       = dataframe.iloc[-1]
            atr        = last.get("atr", current_rate * 0.01)
            atr_ratio  = last.get("atr_ratio", 1.0)
            conf_key   = "conf_score_long" if side == "long" else "conf_score_short"
            confluence = last.get(conf_key, 75)

            # Position size = Risk Amount / (ATR_dist / price)
            # Floor SL 0.5% — konsisten dengan custom_stoploss
            risk_per_unit = max(atr * self.sl_atr_multiplier, current_rate * 0.01)
            base_stake    = risk_amount / (risk_per_unit / current_rate + 1e-10)

            # Volatility adjustment
            if   atr_ratio > 2.0: vol_mult = 0.5
            elif atr_ratio > 1.5: vol_mult = 0.75
            elif atr_ratio < 0.5: vol_mult = 1.25
            else:                  vol_mult = 1.0

            # Confluence adjustment
            if   confluence >= 90: conf_mult = 1.00
            elif confluence >= 75: conf_mult = 0.75
            else:                  conf_mult = 0.50

            # [FIX] min_rrr validation — cek apakah RR tercapai
            # Estimasi TP (2R) vs SL (1R) dari ATR
            sl_pct = risk_per_unit / current_rate
            tp_pct = sl_pct * self.min_rrr
            # Jika ATR ratio terlalu besar, TP mungkin tidak realistis
            if tp_pct > 0.15:  # TP > 15% dari harga → skip
                logger.info(f"[{pair}] Stake rejected: TP {tp_pct:.1%} too large (ATR ratio {atr_ratio:.2f})")
                return min_stake if min_stake else 1.0

            # Streak adjustment
            streak_mult = 1.0
            losses = self.consecutive_losses.get(pair, 0)
            if   losses >= 5: return min_stake if min_stake else 1.0
            elif losses >= 3: streak_mult = 0.50
            elif losses >= 2: streak_mult = 0.75

            final_stake = base_stake * vol_mult * conf_mult * streak_mult
        else:
            final_stake = total_capital * risk_pct

        if SENTIMENT_ENGINE_ENABLED:
            sentiment = _sentiment(pair)
            if not sentiment.get("stale", True):
                fear_greed = sentiment.get("fear_greed")
                if fear_greed is not None and int(fear_greed) <= 20:
                    final_stake *= 0.50
                elif fear_greed is not None and int(fear_greed) <= 35:
                    final_stake *= 0.75

        min_s = min_stake if min_stake else 1.0
        final = max(min_s, min(final_stake, max_stake))

        # ── Micro-account floor ──
        # Dengan modal kecil (<$50), risk-based sizing menghasilkan stake di bawah
        # minNotional exchange ($5) → order akan ditolak Binance. Floor ke margin
        # minimum (minNotional ÷ leverage) agar order bisa jalan tanpa menolak
        # leverage. Contoh: minNotional $5 @ 5x → margin minimum $1.
        try:
            min_notional = 5.0  # Binance USDT-M futures default
            # Coba ambil dari exchange kalau tersedia (get_market sudah ada)
            if hasattr(self, "exchange") and self.exchange:
                mkt = self.exchange.get_market(pair)
                min_notional = float(mkt.get("limits", {}).get("cost", {}).get("min", 5.0))
            # Leverage aktual dari metode leverage() — floor dibagi leverage
            # biar margin minimum konsisten dengan notional minimum exchange.
            try:
                lev = self.leverage(pair, current_time, current_rate,
                                    proposed_leverage=5.0, max_leverage=5.0,
                                    entry_tag=entry_tag, side=side)
                lev = max(lev, 1.0)
            except Exception:
                lev = 1.0
            final = max(final, min_notional / lev)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[{pair}] Min-notional floor skipped: {e}")

        return max(min_s, final)

    # =========================================================================
    # ADAPTIVE LEVERAGE
    # =========================================================================
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float,
                 entry_tag: str | None, side: str) -> float:
        try:
            total_capital = self.wallets.get_total("USDT")
        except Exception:
            total_capital = 100.0

        if   total_capital < 100:  base_lev = 5.0
        elif total_capital < 1000: base_lev = 4.0
        else:                      base_lev = 5.0

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) > 0:
            atr_ratio = dataframe["atr_ratio"].iloc[-1]
            if   atr_ratio > 2.0: base_lev = max(1.0, base_lev * 0.5)
            elif atr_ratio > 1.5: base_lev = max(1.0, base_lev * 0.75)

        return min(base_lev, max_leverage, 5.0)

    # =========================================================================
    # CONFIRM ENTRY — validation pipeline lengkap (v2.4)
    # =========================================================================
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float,
                             rate: float, time_in_force: str,
                             current_time: datetime, entry_tag: str | None,
                             side: str, **kwargs) -> bool:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        # Butuh cukup data untuk indikator matang. 250 candle 5m = 20+ jam —
        # terlalu lama untuk mulai trade. Turunkan ke 100 (8 jam) agar bot
        # mulai lebih cepat, indikator EMA200/ADX sudah cukup stabil.
        if len(dataframe) < 100:
            return False

        last      = dataframe.iloc[-1]
        conf_key  = "conf_score_long" if side == "long" else "conf_score_short"
        conf      = last.get(conf_key, 0)
        atr_ratio = last.get("atr_ratio", 1.0)
        atr       = last.get("atr", rate * 0.01)
        regime = str(last.get("regime", "unknown"))
        factor_score = _quant_factor_score(pair, side, last)
        if factor_score is not None:
            conf = factor_score
        adaptive = _quant_params(pair, regime)
        confluence_threshold = float(adaptive.get("confluence_threshold", self.min_confluence_score))
        sl_multiplier = float(adaptive.get("sl_atr_multiplier", self.sl_atr_multiplier))
        sl_dist   = sl_multiplier * atr
        # Floor SL 0.5% — konsisten dengan custom_stoploss. ATR 5m pair murah
        # bisa sangat kecil, SL terlalu rapat → noise & fee makan RR.
        sl_dist   = max(sl_dist, rate * 0.01)
        sl_pct    = sl_dist / rate

        # ── 1. Confluence check ──
        if conf < confluence_threshold:
            logger.info(f"[{pair}] REJECTED: confluence {conf} < {confluence_threshold}")
            return False

        # ── 1b. ML confirmation check (fail-CLOSED) ──
        # [HARDEN] Fail-closed: kalau model-inference down/error/timeout,
        # entry DIBLOKIR. ML adalah konfirmasi penting; masuk tanpa ML saat
        # market tidak normal = risiko loss-streak (persis akar masalah lama).
        ml_result = _fetch_ml_prediction(pair, dataframe, side, self)
        if ml_result is None:
            logger.info(
                f"[{pair}] REJECTED: ML service unavailable (fail-closed) — "
                f"entry blocked sampai ML respons"
            )
            return False
        ml_signal = str(ml_result.get("signal", "HOLD")).upper()
        ml_prob   = float(ml_result.get("probability", 0.5))
        if side == "long" and ml_signal == "SELL":
            logger.info(
                f"[{pair}] REJECTED: ML signal SELL (prob {ml_prob:.2f}) vs long intent"
            )
            return False
        if side == "short" and ml_signal == "BUY":
            logger.info(
                f"[{pair}] REJECTED: ML signal BUY (prob {ml_prob:.2f}) vs short intent"
            )
            return False

        # News is an additional confidence modifier, never a hard dependency.
        news = _news_classify(pair, os.getenv("NEWS_HEADLINE", ""))
        news_score = float(news.get("score", 0.0) or 0.0)
        if NEWS_ALPHA_ENABLED and ((side == "long" and news_score < -0.75) or
                                   (side == "short" and news_score > 0.75)):
            logger.info("[%s] REJECTED: strongly contradictory news score %.2f", pair, news_score)
            return False

        sentiment = _sentiment(pair)
        if SENTIMENT_ENGINE_ENABLED and not sentiment.get("stale", True):
            fear_greed = sentiment.get("fear_greed")
            sentiment_score = float(sentiment.get("score", 0.0) or 0.0)
            if fear_greed is not None and int(fear_greed) >= 80:
                logger.info("[%s] REJECTED: extreme greed Fear & Greed=%s", pair, fear_greed)
                return False
            if ((side == "long" and sentiment_score <= -0.35) or
                    (side == "short" and sentiment_score >= 0.35)):
                logger.info("[%s] REJECTED: contradictory sentiment score %.2f", pair, sentiment_score)
                return False

        # Funding/OI is an optional on-chain confirmation. Extreme positive
        # funding means crowded longs; extreme negative funding crowds shorts.
        chain = _onchain_metrics(pair)
        funding = float(chain.get("aggregated_funding_rate", 0.0) or 0.0)
        oi_delta = float(chain.get("open_interest_delta_pct", 0.0) or 0.0)
        netflow_signal = chain.get("netflow_signal")
        if ON_CHAIN_ENGINE_ENABLED and ((side == "long" and funding >= .001) or
                                        (side == "short" and funding <= -.001)):
            logger.info("[%s] REJECTED: crowded funding %.5f", pair, funding)
            return False
        if ON_CHAIN_ENGINE_ENABLED and abs(oi_delta) >= 0.05 and ((side == "long" and funding > 0) or
                                                                  (side == "short" and funding < 0)):
            logger.info("[%s] REJECTED: crowded funding/OI expansion funding=%.5f oi_delta=%.2f%%",
                        pair, funding, oi_delta * 100)
            return False
        if ON_CHAIN_ENGINE_ENABLED and netflow_signal is not None:
            try:
                netflow_signal = float(netflow_signal)
                if ((side == "long" and netflow_signal <= -0.75) or
                        (side == "short" and netflow_signal >= 0.75)):
                    logger.info("[%s] REJECTED: contradictory exchange netflow signal %.2f",
                                pair, netflow_signal)
                    return False
            except (TypeError, ValueError):
                pass

        # ── 2. Dead zone & FOMO lock ──
        if last.get("session_dead", 0) == 1:
            return False
        positive_news_override = (
            side == "long" and NEWS_ALPHA_ENABLED and news_score >= 0.75 and
            news.get("event") == "positive_event"
        )
        if last.get("fomo_lock", 0) == 1 and not positive_news_override:
            return False

        # ── 3. Extreme volatility ──
        if atr_ratio > 3.0:
            logger.info(f"[{pair}] REJECTED: ATR ratio {atr_ratio:.2f} (extreme)")
            return False

        # ── 4. Consecutive loss streak ──
        if self.consecutive_losses.get(pair, 0) >= 5:
            logger.warning(f"[{pair}] REJECTED: 5 consecutive losses — strategy halted")
            return False

        # ── 4b. [HARDEN] GLOBAL loss breaker (harian) ──
        # Hitung loss beruntun HARI INI (00:00 UTC s.d. sekarang) dari DB.
        # Otomatis reset ke 0 setiap pergantian hari UTC (mencegah deadlock).
        try:
            today_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
            closed_today = Trade.get_trades([
                Trade.close_date >= today_start,
                Trade.is_open.is_(False),
            ]).all()
            consecutive_today = 0
            for t in sorted(closed_today, key=lambda x: x.close_date or x.close_date_utc, reverse=True):
                if t.close_profit_abs is not None and t.close_profit_abs < 0:
                    consecutive_today += 1
                else:
                    break
            if consecutive_today >= self.max_consecutive_losses_global:
                logger.warning(
                    f"[{pair}] REJECTED: GLOBAL consecutive-loss breaker aktif "
                    f"({consecutive_today} losses hari ini) — entry blocked"
                )
                return False
        except Exception as exc:
            logger.debug(f"[{pair}] Global loss breaker check skipped: {exc}")

        # ── 5. Fee-aware RR check ──
        # Round-trip fee = 2x taker fee (entry + exit)
        fee_cost = 2 * EXCHANGE_TAKER_FEE
        if side == "long":
            potential_tp = last.get("r2", rate * (1 + sl_pct * self.min_rrr))
        else:
            potential_tp = last.get("s2", rate * (1 - sl_pct * self.min_rrr))
        gross_rr = abs(potential_tp - rate) / (sl_dist + 1e-10)
        # RR efektif setelah fee (SL juga kena fee, jadi net_sl_pct += fee)
        net_sl_pct = sl_pct + fee_cost
        net_tp_pct = abs(potential_tp - rate) / rate - fee_cost
        estimated_rr = net_tp_pct / (net_sl_pct + 1e-10)
        if estimated_rr < self.min_rrr:
            logger.info(
                f"[{pair}] REJECTED: fee-adjusted RR {estimated_rr:.2f} < {self.min_rrr} "
                f"(gross RR {gross_rr:.2f}, fee {fee_cost:.3%})"
            )
            return False

        # ── 5b. [RISK-CAP] Structure-implied SL cap ──
        # Hitung implied SL dari struktur swing (SAMA dgn custom_stoploss L2).
        # Kalau struktur terlalu jauh → implied SL > cap → setup tak cocok utk
        # ukuran akun ini. Tolak DI SINI, jangan buka posisi lalu pasang SL lebar
        # (akar anomali MUBARAK/ACU). Emergency-loosen (×1.5 restart) beda konteks
        # — itu darurat teknis di custom_stoploss, di-handle terpisah + alert.
        struct_pad = max(sl_dist * 0.35, rate * 0.0025)
        if side == "long":
            swing = last.get("swing_low5")
            implied_sl_dist = (rate - (float(swing) - struct_pad)) if swing is not None and not pd.isna(swing) else sl_dist
        else:
            swing = last.get("swing_high5")
            implied_sl_dist = ((float(swing) + struct_pad) - rate) if swing is not None and not pd.isna(swing) else sl_dist
        implied_sl_pct = max(implied_sl_dist, sl_dist) / rate
        if implied_sl_pct > self.max_entry_sl_pct:
            logger.info(
                f"[{pair}] REJECTED: implied SL {implied_sl_pct:.2%} > cap "
                f"{self.max_entry_sl_pct:.2%} (struktur swing terlalu jauh — "
                f"setup tak cocok utk ukuran akun)"
            )
            return False

        # ── 6. Liquidation price check ──
        # Pastikan SL berada SEBELUM harga likuidasi (leverage aware)
        try:
            leverage = self.leverage(
                pair=pair, current_time=current_time, current_rate=rate,
                proposed_leverage=3.0, max_leverage=5.0,
                entry_tag=entry_tag, side=side
            )
            if side == "long":
                liq_price = rate * (1 - 1 / leverage)  # Approx liq price for long
                sl_price  = rate - sl_dist
                if sl_price <= liq_price:
                    logger.warning(
                        f"[{pair}] REJECTED: SL {sl_price:.4f} at or below liq price "
                        f"{liq_price:.4f} (lev {leverage}x)"
                    )
                    return False
            else:
                liq_price = rate * (1 + 1 / leverage)  # Approx liq price for short
                sl_price  = rate + sl_dist
                if sl_price >= liq_price:
                    logger.warning(
                        f"[{pair}] REJECTED: SL {sl_price:.4f} at or above liq price "
                        f"{liq_price:.4f} (lev {leverage}x)"
                    )
                    return False
        except Exception as e:
            logger.debug(f"[{pair}] Liquidation check skipped: {e}")

        # ── 7. Real funding rate check via Exchange API ──
        try:
            # Freqtrade DataProvider tidak expose funding rate secara langsung.
            # Gunakan self.exchange jika tersedia, fallback ke ATR% proxy.
            if hasattr(self, 'exchange') and self.exchange:
                funding_data = self.exchange.fetch_funding_rate(pair)
                funding_rate = funding_data.get("fundingRate", 0.0) if funding_data else 0.0
                # funding rate > 0.05% per 8h = cost signifikan untuk long
                if side == "long" and funding_rate > 0.0005:
                    logger.info(
                        f"[{pair}] REJECTED long: funding rate {funding_rate:.4%} too high (cost)"
                    )
                    return False
                if side == "short" and funding_rate < -0.0005:
                    logger.info(
                        f"[{pair}] REJECTED short: funding rate {funding_rate:.4%} negative (cost)"
                    )
                    return False
            else:
                raise AttributeError("exchange not available")
        except Exception:
            # Fallback: pakai ATR% proxy jika exchange API tidak tersedia
            atr_pct = last.get("atr_pct", 1.0)
            if atr_pct < 0.3:
                logger.info(
                    f"[{pair}] REJECTED: ATR {atr_pct:.2f}% too small vs estimated funding"
                )
                return False

        # ── 8. Computed correlation guard (Supreme Math Edition) ──
        # Hitung Pearson correlation real-time antara candidate pair dan open positions
        try:
            open_trades = Trade.get_trades([Trade.is_open.is_(True)]).all()
            if open_trades:
                cand_df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                cand_returns = cand_df["close"].pct_change().dropna().tail(50).tolist() if len(cand_df) >= 20 else []

                high_corr_found = False
                for t in open_trades:
                    if t.pair == pair:
                        continue
                    t_df, _ = self.dp.get_analyzed_dataframe(t.pair, self.timeframe)
                    t_returns = t_df["close"].pct_change().dropna().tail(50).tolist() if len(t_df) >= 20 else []
                    if cand_returns and t_returns:
                        corr_val = pearson(cand_returns, t_returns)
                        if corr_val >= 0.85:
                            logger.info(
                                f"[{pair}] REJECTED: high computed correlation r={corr_val:.2f} "
                                f"with open position {t.pair} (max 0.85)"
                            )
                            return False
                    else:
                        # Fallback ke cluster btc jika data return kurang
                        btc_cluster = {"BTC", "ETH", "BNB", "SOL", "XRP"}
                        b_cand = pair.split("/")[0].split(":")[0]
                        b_open = t.pair.split("/")[0].split(":")[0]
                        if b_cand in btc_cluster and b_open in btc_cluster:
                            high_corr_found = True

                if high_corr_found:
                    correlated_count = sum(
                        1 for t in open_trades
                        if t.pair.split("/")[0].split(":")[0] in {"BTC", "ETH", "BNB", "SOL", "XRP"}
                    )
                    if correlated_count >= self.max_correlated_open:
                        logger.info(
                            f"[{pair}] REJECTED: {correlated_count} correlated cluster positions open "
                            f"(max {self.max_correlated_open})"
                        )
                        return False
        except Exception as e:
            logger.debug(f"[{pair}] Correlation check skipped: {e}")

        logger.info(
            f"[{pair}] CONFIRMED: side={side}, conf={conf:.0f}, "
            f"atr_ratio={atr_ratio:.2f}, fee_adj_rr={estimated_rr:.2f}"
        )

        # ── [ORDERBOOK-GATE] Imbalance is an optional data-driven gate ──
        # I(t) = (B-A)/(B+A). It is always recorded; rejection is controlled
        # by ORDERBOOK_GATE_ENABLED and the configured threshold.
        try:
            book = self.dp.orderbook(pair, 10)
            if book and book.get("bids") and book.get("asks"):
                bid_vol = sum(float(b[1]) for b in book["bids"])
                ask_vol = sum(float(a[1]) for a in book["asks"])
                imb = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)
                # Untuk long kita mau imb>0 (bid dominan), short imb<0
                side_ok = (side == "long" and imb > 0) or (side == "short" and imb < 0)
                logger.info(
                    f"[ORDERBOOK-PROBE] [{pair}] I(t)={imb:+.3f} "
                    f"({'side-aligned' if side_ok else 'CONTRA'}) | bids={bid_vol:.2f} asks={ask_vol:.2f}"
                )
                # [LINK-OUTCOME] Simpan sementara → di-persist ke CustomDataWrapper
                # saat custom_stoploss after_fill (trade.id baru ADA di situ).
                # max_open_trades=1 → aman 1 pending aja. ponytail: upgrade ke
                # dict per-pair kalau max_open_trades>1.
                self._pending_imbalance = {"imb": float(imb), "ts": str(current_time)}
                if os.getenv("ORDERBOOK_GATE_ENABLED", "false").lower() == "true" and abs(imb) >= float(os.getenv("ORDERBOOK_GATE_MIN_IMBALANCE", "0.08")) and not side_ok:
                    logger.info(f"[{pair}] REJECTED: orderbook imbalance contra intent ({imb:+.3f})")
                    return False
                intelligence = _orderbook_intelligence(pair, book)
                if intelligence:
                    self._pending_orderbook = intelligence
                    spoofing = float(intelligence.get("spoofing_score", 0.0) or 0.0)
                    if (os.getenv("ORDERBOOK_GATE_ENABLED", "false").lower() == "true" and
                            spoofing > float(os.getenv("ORDERBOOK_SPOOFING_MAX", "0.70"))):
                        logger.info("[%s] REJECTED: orderbook spoofing score %.2f", pair, spoofing)
                        return False
        except Exception as e:
            logger.debug(f"[{pair}] orderbook probe skipped: {e}")

        # ── 8. Risk Gateway validation (fail-closed) ──
        try:
            import aiohttp
            import asyncio as _asyncio
            RISK_GATEWAY_URL = os.getenv("RISK_GATEWAY_URL", "http://risk-gateway:8000")
            _trade_id = f"{pair}_{current_time.strftime('%Y%m%d%H%M%S')}"
            _side = "buy" if side == "long" else "sell"
            intent = {
                "trade_id": _trade_id,
                "client_order_id": _trade_id,
                "strategy_version": "AITradingStrategy",
                "config_version": "freqtrade",
                "pair": pair.split("/")[0].replace(":", "") + "USDT",
                "side": _side,
                "order_type": "limit",
                "amount": float(amount),
                "price": float(rate),
                "leverage": 3,
                "margin_mode": "isolated",
                "stop_loss": float(rate - sl_dist) if side == "long" else float(rate + sl_dist),
                "strategy_version": "AITradingStrategy",
            }
            try:
                loop = _asyncio.get_event_loop()
            except RuntimeError:
                loop = _asyncio.new_event_loop()
                _asyncio.set_event_loop(loop)

            async def _validate():
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{RISK_GATEWAY_URL}/validate",
                        json=intent,
                        timeout=aiohttp.ClientTimeout(total=3)
                    ) as resp:
                        if resp.status == 200:
                            result = await resp.json()
                            decision = result.get("decision", "rejected")
                            if decision != "approved":
                                reason = result.get("reason", "unknown")
                                logger.info(f"[{pair}] REJECTED by risk-gateway: {reason}")
                                return False
                            logger.info(f"[{pair}] APPROVED by risk-gateway ✅")
                            return True
                        else:
                            logger.warning(f"[{pair}] risk-gateway returned {resp.status} - FAIL CLOSED")
                            return False

            try:
                risk_approved = loop.run_until_complete(_validate())
                if not risk_approved:
                    return False
            except Exception as e:
                logger.warning(f"[{pair}] risk-gateway error: {e} - FAIL CLOSED")
                return False
        except Exception as e:
            logger.warning(f"[{pair}] risk-gateway setup error: {e} - FAIL CLOSED")
            return False

        return True

    # =========================================================================
    # [FASE 1 FEEDBACK LOOP] order_filled — dual-write trade outcome saat exit
    # Freqtrade 2026.3: order_filled(self, pair, trade, order, current_time)
    # dipanggil setiap kali sebuah order TERISI (entry maupun exit).
    # Kita hanya emit saat trade sudah benar-benar tertutup (exit fill terakhir).
    # =========================================================================
    def order_filled(self, pair: str, trade, order, current_time: datetime,
                     **kwargs) -> None:
        try:
            # Entry fill adalah titik pertama ketika objek trade sudah
            # persisten. Snapshot kondisi entry disimpan di sini, bukan di
            # confirm_trade_entry yang memang tidak menerima objek trade.
            if trade.is_open:
                if not getattr(trade, "fb_entry", None):
                    dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                    if len(dataframe):
                        last = dataframe.iloc[-1]
                        feature_cols = [
                            "ema8", "ema13", "ema21", "ema34", "ema50", "ema89", "ema200",
                            "sma20", "sma50", "sma200", "rsi_14", "rsi_7", "rsi_21",
                            "macd", "macd_signal", "macd_hist", "stoch_k", "stoch_d",
                            "srsi_k", "srsi_d", "cci", "willr", "mfi", "roc", "atr",
                            "atr_pct", "atr_ratio", "bb_width", "bb_pct", "bb_squeeze",
                            "kc_upper", "kc_lower", "don_high", "don_low", "obv_slope",
                            "volume_ratio", "volume_slope", "cmf", "vwma", "adx",
                            "ema50_1d", "ema200_1d", "ema50_4h", "ema200_4h", "adx_1d",
                            "adx_4h", "rsi_4h", "btc_ema50_1h", "btc_rsi_1h", "mtf_alignment",
                        ]
                        features = {
                            name: float(last[name]) for name in feature_cols
                            if name in last.index and isinstance(last[name], (int, float))
                            and not pd.isna(last[name])
                        }
                        features.update({"feature_version": "v1", "candle_count": int(len(dataframe))})
                        regime = str(last.get("regime", "RANGING"))
                        from shared.feedback import snapshot_entry_conditions
                        snapshot_entry_conditions(
                            trade=trade,
                            regime=regime,
                            predicted_rr=float(last.get("predicted_rr", self.tp3_rrr) or self.tp3_rrr),
                            ml_signal=str(last.get("ml_signal_direction", "HOLD")),
                            ml_prob=float(last.get("ml_probability", 0.5) or 0.5),
                            conf=float(last.get(
                                "conf_score_short" if trade.is_short else "conf_score_long", 0
                            ) or 0),
                            atr_ratio=float(last.get("atr_ratio", 1.0) or 1.0),
                            side="short" if trade.is_short else "long",
                            entry_rate=float(trade.open_rate),
                            features=features,
                        )
                        trade.entry_regime = regime
                return

            # Hanya proses close penuh (bukan entry / partial fill)
            fb = getattr(trade, "fb_entry", {}) or {}
            side = "short" if trade.is_short else "long"
            entry_rate = float(trade.open_rate or fb.get("entry_rate") or 0.0)
            exit_rate = float(trade.close_rate or 0.0)

            # actual_rr: |profit%| / |risk%| di mana risk = predicted SL distance
            predicted_rr = fb.get("predicted_rr")
            pnl_pct = float(trade.close_profit or 0.0)
            actual_rr = None
            try:
                sl_price = (getattr(trade, "ml_sl_tp", {}) or {}).get("sl_price")
                if sl_price and entry_rate > 0:
                    risk_pct = abs(entry_rate - float(sl_price)) / entry_rate
                    if risk_pct > 1e-9:
                        actual_rr = abs(pnl_pct) / risk_pct * (1 if pnl_pct >= 0 else -1)
            except Exception:
                actual_rr = None

            outcome = {
                "trade_id": int(trade.id),
                "pair": pair,
                "timeframe": self.timeframe,
                "entry_conditions": {
                    "regime": fb.get("regime"),
                    "predicted_rr": predicted_rr,
                    "ml_signal": fb.get("ml_signal"),
                    "ml_prob": fb.get("ml_prob"),
                    "conf_score": fb.get("conf_score"),
                    "atr_ratio": fb.get("atr_ratio"),
                    "side": side,
                    "entry_rate": entry_rate,
                    "exit_rate": exit_rate,
                    "features": fb.get("features") or {},
                },
                "exit_reason": str(trade.exit_reason or "unknown"),
                "pnl_pct": pnl_pct,
                "pnl_abs": float(trade.close_profit_abs or 0.0),
                "predicted_rr": predicted_rr,
                "actual_rr": actual_rr,
                "regime_at_entry": fb.get("regime"),
                "timestamp_entry": trade.open_date_utc.isoformat() if trade.open_date_utc else current_time.isoformat(),
                "timestamp_exit": (trade.close_date_utc or current_time).isoformat(),
            }

            # Dual-write async tanpa blocking trading loop (fire-and-forget).
            import asyncio as _asyncio
            from shared.feedback import emit_trade_closed
            try:
                loop = _asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(emit_trade_closed(outcome))
                else:
                    loop.run_until_complete(emit_trade_closed(outcome))
            except RuntimeError:
                _asyncio.run(emit_trade_closed(outcome))
            logger.info(f"[{pair}] feedback loop: TRADE_CLOSED emitted (trade_id={trade.id})")
        except Exception as e:
            # WAJIB non-fatal: kegagalan feedback loop tidak boleh crash strategy
            logger.warning(f"[{pair}] order_filled feedback emit failed: {e}")

    # =========================================================================
    # [FIX-MAJOR] STREAK TRACKER — via confirm_trade_exit (hook resmi Freqtrade v3)
    # order_filled() BUKAN interface Freqtrade → tidak pernah terpanggil otomatis.
    # confirm_trade_exit() dipanggil setiap kali trade akan ditutup.
    # =========================================================================
    def confirm_trade_exit(self, pair: str, trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        """
        Dipanggil Freqtrade sebelum order exit dieksekusi.
        Digunakan untuk update streak tracker & cleanup partial TP state.
        Selalu return True (confirm exit), kecuali ada alasan khusus.
        """
        # Estimasi profit dari rate exit vs open_rate
        # (close_profit belum tersedia saat hook ini dipanggil)
        if trade.open_rate and rate:
            if trade.is_short:
                est_profit = (trade.open_rate - rate) / trade.open_rate
            else:
                est_profit = (rate - trade.open_rate) / trade.open_rate

            if est_profit < 0:
                self.consecutive_losses[pair] = self.consecutive_losses.get(pair, 0) + 1
                if self.consecutive_losses[pair] >= 3:
                    logger.critical(
                        f"[{pair}] AUTO-PAUSE: {self.consecutive_losses[pair]} consecutive losses"
                    )
            else:
                self.consecutive_losses[pair] = 0

        # Cleanup partial TP state untuk trade ini
        if trade.id in self.partial_tp_done:
            del self.partial_tp_done[trade.id]

        return True  # Selalu izinkan exit

    # =========================================================================
    # INFORMATIVE PAIRS
    # =========================================================================
    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative = [(p, tf) for p in pairs for tf in ["15m", "1h", "4h", "1d"]]
        # [NEW] BTC macro filter: tambahkan BTC/USDT:USDT 1h untuk semua pair
        informative.append(("BTC/USDT:USDT", "1h"))
        return informative


# =============================================================================
# [NEW] CUSTOM HYPEROPT LOSS — Calmar Ratio
# Calmar = Total Profit / Max Drawdown → lebih relevan untuk futures daripada Sharpe
# Penalti berat pada drawdown besar, reward untuk trade konsisten profit.
# =============================================================================
class CalmarRatioHyperOptLoss(IHyperOptLoss):
    """
    Custom HyperOpt Loss Function berbasis Calmar Ratio.
    Calmar Ratio = Annualized Return / Max Drawdown

    Cocok untuk futures karena:
    - Penalti sangat berat pada drawdown besar (capital preservation)
    - Reward untuk strategi dengan equity curve smooth
    - Tidak ter-bias oleh volatility tinggi seperti Sharpe

    Cara pakai:
        freqtrade hyperopt --hyperopt-loss CalmarRatioHyperOptLoss ...
    """

    @staticmethod
    def hyperopt_loss_function(results: DataFrame, trade_count: int,
                               min_date: datetime, max_date: datetime,
                               *args, **kwargs) -> float:
        if trade_count < 10:
            return 1.0  # Terlalu sedikit trade → penalty

        total_profit = results["profit_abs"].sum()
        if total_profit <= 0:
            return 1.0  # Tidak profitable

        # Hitung max drawdown dari equity curve
        equity = results["profit_abs"].cumsum()
        peak   = equity.cummax()
        dd     = (equity - peak)
        max_dd = abs(dd.min())

        if max_dd < 1e-6:
            max_dd = 1e-6  # Avoid division by zero

        # Annualization factor (365 hari)
        days = max((max_date - min_date).days, 1)
        annual_factor = 365.0 / days
        annualized_return = total_profit * annual_factor

        calmar = annualized_return / max_dd

        # Penalty jika trade count terlalu sedikit (< 1 trade/day)
        trades_per_day = trade_count / days
        if trades_per_day < 0.5:
            calmar *= 0.5

        # Return nilai negatif (HyperOpt minimize loss)
        return -calmar


# =============================================================================
# CONFIG TEMPLATE
# =============================================================================
CONFIG_TEMPLATE = {
    "max_open_trades": 3,
    "stake_currency": "USDT",
    "stake_amount": "unlimited",
    "tradable_balance_ratio": 0.20,
    "fiat_display_currency": "USD",
    "timeframe": "5m",
    "dry_run": True,
    "dry_run_wallet": 100,
    "cancel_open_orders_on_exit": True,
    "trading_mode": "futures",
    "margin_mode": "isolated",
    "minimal_roi": {"0": 0.50},
    "stoploss": -0.99,
    "use_custom_stoploss": True,
    "trailing_stop": False,
    "trailing_stop_positive": 0.004,
    "trailing_stop_positive_offset": 0.006,
    "trailing_only_offset_is_reached": True,
    "unfilledtimeout": {"entry": 5, "exit": 15, "exit_timeout_count": 3, "unit": "minutes"},
    "exchange": {
        "name": "binance",
        "key": "YOUR_API_KEY",
        "secret": "YOUR_SECRET_KEY",
        "ccxt_config": {"options": {"defaultType": "future"}}
    },
    "telegram": {"enabled": False, "token": "YOUR_TOKEN", "chat_id": "YOUR_CHAT_ID"},
    "strategy": "AITradingStrategy",
    "initial_state": "running",
    "force_entry_enable": False,
    "internals": {"process_throttle_secs": 5}
}


if __name__ == "__main__":
    import json
    print("AI Trading Strategy v2.4 - Fully Enhanced & Ready")
    print("\n=== v2.2 Fixes ===")
    print("  [CRITICAL] Regime detection vectorized (no lookahead bias)")
    print("  [CRITICAL] ConfluenceScorer now uses real MTF informative columns")
    print("  [MAJOR]    Entry conditions: 3 hard gates + confluence score")
    print("  [MAJOR]    Partial TP via adjust_trade_position() (TP1/TP2/TP3 working)")
    print("  [MINOR]    SL sign + min_rrr + funding proxy fixed")
    print("\n=== v2.3 Bug Fixes ===")
    print("  [CRITICAL] SL trail logic order fixed (2R before 1R)")
    print("  [MAJOR]    confirm_trade_exit() for streak tracking")
    print("  [MAJOR]    Volume cross-timeframe fixed, RSI short range fixed")
    print("  [MINOR]    minimal_roi safety net, dead vars removed")
    print("\n=== v2.4 New Features ===")
    print("  [CRITICAL] Real daily P&L circuit breaker (Trade persistence)")
    print("  [CRITICAL] Liquidation price check before entry")
    print("  [MAJOR]    BTC macro filter @informative (altcoin aligned to BTC)")
    print("  [MAJOR]    Freqtrade protections (CooldownPeriod + StoplossGuard + MaxDrawdown)")
    print("  [MAJOR]    OBV slope + VWAP in confluence score")
    print("  [MAJOR]    Max trade age timeout (24h flat exit)")
    print("  [MINOR]    Fee-aware RR (0.08% round-trip included)")
    print("  [MINOR]    Real funding rate via exchange.fetch_funding_rate() + ATR fallback")
    print("  [MINOR]    Correlation guard (max 2 correlated majors open)")
    print("  [MINOR]    CalmarRatioHyperOptLoss for hyperopt")
    print("\nConfig template:")
    print(json.dumps(CONFIG_TEMPLATE, indent=2))
