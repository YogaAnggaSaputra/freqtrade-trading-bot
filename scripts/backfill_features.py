#!/usr/bin/env python3
"""Backfill fitur entry ke trade_outcomes dari market_candles (Postgres).

Rekonstruksi fitur 5m (sama dengan _FEATURE_COLS forward di strategy)
untuk trade lama yang trade_outcomes.entry_conditions.features masih
kosong. Anti look-ahead: hanya candle dengan timestamp <= open_date
trade yang dipakai; fitur dihitung dari candle TSB (close == candle
terakhir <= entry).

Usage: python backfill_features.py [--dry-run] [--limit N]
"""
import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone

import asyncpg
import numpy as np
import pandas as pd

# fitur 5m-only, konsisten dengan _FEATURE_COLS forward
FEATURE_COLS_5M = [
    "ema8", "ema13", "ema21", "ema34", "ema50", "ema89", "ema200",
    "sma20", "sma50", "sma200", "rsi_14", "rsi_7", "rsi_21",
    "macd", "macd_signal", "macd_hist", "stoch_k", "stoch_d",
    "srsi_k", "srsi_d", "cci", "willr", "mfi", "roc", "atr",
    "atr_pct", "atr_ratio", "bb_width", "bb_pct", "bb_squeeze",
    "kc_upper", "kc_lower", "don_high", "don_low", "obv_slope",
    "volume_ratio", "volume_slope", "cmf", "vwma", "adx",
]

MIN_CANDLES = 60  # butuh warmup indikator (ema200 = 200 candle)


def compute_features(candles: pd.DataFrame) -> dict:
    """Hitung fitur 5m identik dengan populate_indicators strategy.

    candles: DataFrame [open, high, low, close, volume] urut lama→baru.
    """
    close = candles["close"]
    high = candles["high"]
    low = candles["low"]
    vol = candles["volume"]
    open_ = candles["open"]
    f: dict = {}

    # EMA
    for p in [8, 13, 21, 34, 50, 89, 144, 200]:
        f[f"ema{p}"] = float(close.ewm(span=p, adjust=False).mean().iloc[-1])
    # SMA
    for p in [20, 50, 200]:
        f[f"sma{p}"] = float(close.rolling(p).mean().iloc[-1]) if len(close) >= p else float("nan")

    # RSI (Wilder)
    def _rsi(s: pd.Series, period: int) -> float:
        d = s.diff()
        gain = d.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
        loss = (-d.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
        rs = gain / (loss + 1e-10)
        return float(100 - 100 / (1 + rs).iloc[-1]) if len(s) > period else float("nan")

    f["rsi_14"] = _rsi(close, 14)
    f["rsi_7"] = _rsi(close, 7)
    f["rsi_21"] = _rsi(close, 21)
    rsi14_s = close.diff().clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean() / (
        (-close.diff().clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean() + 1e-10
    )
    rsi14_s = 100 - 100 / (1 + rsi14_s)

    # MACD (12,26,9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False).mean()
    f["macd"] = float(macd.iloc[-1])
    f["macd_signal"] = float(sig.iloc[-1])
    f["macd_hist"] = float((macd - sig).iloc[-1])

    # Stochastic %K/%D (14,3,3)
    ll = low.rolling(14).min()
    hh = high.rolling(14).max()
    stoch_k = 100 * (close - ll) / (hh - ll + 1e-10)
    f["stoch_k"] = float(stoch_k.iloc[-1])
    f["stoch_d"] = float(stoch_k.rolling(3).mean().iloc[-1])

    # Stochastic RSI (14,5,3)
    srsi_min = rsi14_s.rolling(5).min()
    srsi_max = rsi14_s.rolling(5).max()
    srsi_k = 100 * (rsi14_s - srsi_min) / (srsi_max - srsi_min + 1e-10)
    f["srsi_k"] = float(srsi_k.iloc[-1])
    f["srsi_d"] = float(srsi_k.rolling(3).mean().iloc[-1])

    # CCI (14)
    tp = (high + low + close) / 3
    cci = (tp - tp.rolling(14).mean()) / (1.5 * tp.rolling(14).std(ddof=0) + 1e-10)
    f["cci"] = float(cci.iloc[-1])

    hh = high.rolling(14).max()
    # Williams %R (14) — butuh aligned rolling
    willr_series = -100 * (hh - close) / (hh - low.rolling(14).min() + 1e-10)
    f["willr"] = float(willr_series.iloc[-1]) if len(willr_series) >= 14 else float("nan")

    # MFI (14)
    mf = (close - low) - (high - close)
    mf_pos = (mf * vol).clip(lower=0).rolling(14).sum()
    mf_neg = (-mf * vol).clip(lower=0).rolling(14).sum()
    mfi = 100 - 100 / (1 + mf_pos / (mf_neg + 1e-10))
    f["mfi"] = float(mfi.iloc[-1])

    # ROC (10)
    f["roc"] = float(close.pct_change(10).iloc[-1] * 100) if len(close) > 10 else float("nan")

    # ATR (14, Wilder)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    f["atr"] = float(atr.iloc[-1])
    f["atr_pct"] = float(atr.iloc[-1] / close.iloc[-1] * 100)
    f["atr_ratio"] = float(atr.iloc[-1] / atr.rolling(20).mean().iloc[-1] + 1e-10) if len(atr) >= 20 else float("nan")

    # Bollinger (20,2)
    bb_m = close.rolling(20).mean()
    bb_s = close.rolling(20).std(ddof=0)
    bb_u = bb_m + 2 * bb_s
    bb_l = bb_m - 2 * bb_s
    f["bb_width"] = float(((bb_u - bb_l) / bb_m).iloc[-1])
    f["bb_pct"] = float(((close - bb_l) / (bb_u - bb_l + 1e-10)).iloc[-1])

    # BB squeeze: BB inside Keltner (20,1.5)
    kc_atr = tr.ewm(alpha=1 / 20, adjust=False).mean()
    kc_mid = close.ewm(span=20, adjust=False).mean()
    kc_u = kc_mid + 1.5 * kc_atr
    kc_l = kc_mid - 1.5 * kc_atr
    f["kc_upper"] = float(kc_u.iloc[-1])
    f["kc_lower"] = float(kc_l.iloc[-1])
    f["bb_squeeze"] = float(1 if (bb_u.iloc[-1] < kc_u.iloc[-1] and bb_l.iloc[-1] > kc_l.iloc[-1]) else 0)

    # Donchian (20)
    f["don_high"] = float(high.rolling(20).max().iloc[-1])
    f["don_low"] = float(low.rolling(20).min().iloc[-1])

    # OBV slope (5)
    obv = (np.sign(close.diff().fillna(0)) * vol).cumsum()
    f["obv_slope"] = float(obv.pct_change(5).iloc[-1]) if len(obv) > 5 else float("nan")

    # Volume
    f["volume_ratio"] = float(vol.iloc[-1] / (vol.rolling(20).mean().iloc[-1] + 1e-10))
    f["volume_slope"] = float(vol.pct_change(5).iloc[-1]) if len(vol) > 5 else float("nan")

    # CMF (20)
    mf_mult = ((close - low) - (high - close)) / (high - low + 1e-10)
    f["cmf"] = float((mf_mult * vol).rolling(20).sum().iloc[-1] / (vol.rolling(20).sum().iloc[-1] + 1e-10))

    # VWMA (20)
    f["vwma"] = float((close * vol).rolling(20).sum().iloc[-1] / (vol.rolling(20).sum().iloc[-1] + 1e-10))

    # ADX (14)
    up = high.diff()
    dn = -low.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr14 = tr.ewm(alpha=1 / 14, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1 / 14, adjust=False).mean() / (tr14 + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1 / 14, adjust=False).mean() / (tr14 + 1e-10)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(alpha=1 / 14, adjust=False).mean()
    f["adx"] = float(adx.iloc[-1])

    return {k: v for k, v in f.items() if isinstance(v, (int, float)) and not np.isnan(v)}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="jangan tulis DB, cuma hitung")
    ap.add_argument("--limit", type=int, default=0, help="batasi jumlah trade (0 = semua)")
    args = ap.parse_args()

    pw = os.getenv("DB_PASSWORD", "changeme")
    pool = await asyncpg.create_pool(
        host=os.getenv("DB_HOST", "postgres"),
        port=int(os.getenv("DB_PORT", "5432")),
        user=os.getenv("DB_USER", "botbinance"),
        password=pw,
        database=os.getenv("DB_NAME", "botbinance"),
        min_size=1, max_size=3,
    )

    # 1. ambil outcome yg belum punya fitur
    rows = await pool.fetch("""
        SELECT trade_id, pair, entry_conditions, timestamp_entry, timestamp_exit
        FROM trade_outcomes
        WHERE entry_conditions IS NULL
           OR NOT entry_conditions ? 'features'
           OR jsonb_typeof(entry_conditions->'features') IS DISTINCT FROM 'object'
           OR (SELECT count(*) FROM jsonb_object_keys(entry_conditions->'features')) < 5
        ORDER BY timestamp_entry
    """)
    if args.limit:
        rows = rows[: args.limit]
    print(f"outcome tanpa fitur: {len(rows)}", flush=True)
    if not rows:
        return

    # 2. per trade: ambil candle 5m dari market_candles (hanya <= entry ts)
    updated = 0
    skipped = 0
    for r in rows:
        trade_id = r["trade_id"]
        pair = r["pair"]
        ts_entry = r["timestamp_entry"]
        if isinstance(ts_entry, datetime):
            ts_entry = ts_entry.replace(tzinfo=None) if ts_entry.tzinfo else ts_entry

        candles = await pool.fetch("""
            SELECT open, high, low, close, volume
            FROM market_candles
            WHERE pair = $1 AND timeframe = '5m' AND timestamp <= $2
            ORDER BY timestamp DESC
            LIMIT 400
        """, pair, ts_entry)
        if len(candles) < MIN_CANDLES:
            skipped += 1
            print(f"  [{trade_id}] {pair}: SKIP (candle {len(candles)} < {MIN_CANDLES})", flush=True)
            continue

        df = pd.DataFrame([dict(c) for c in reversed(candles)],
                          columns=["open", "high", "low", "close", "volume"]).astype(float)
        feats = compute_features(df)
        if len(feats) < 5:
            skipped += 1
            print(f"  [{trade_id}] {pair}: SKIP (fitur {len(feats)} < 5)", flush=True)
            continue
        feats["feature_version"] = "v1"
        feats["candle_count"] = len(df)

        ec_raw = r["entry_conditions"]
        if isinstance(ec_raw, str):
            ec = json.loads(ec_raw or "{}")
        elif isinstance(ec_raw, dict):
            ec = dict(ec_raw)
        else:
            ec = {}
        ec["features"] = feats
        ec["regime"] = ec.get("regime") or "RANGING"

        if args.dry_run:
            print(f"  [{trade_id}] {pair}: {len(feats)} fitur (DRY)", flush=True)
        else:
            await pool.execute(
                "UPDATE trade_outcomes SET entry_conditions = $1 WHERE trade_id = $2",
                json.dumps(ec, default=str), trade_id,
            )
            print(f"  [{trade_id}] {pair}: UPDATED {len(feats)} fitur", flush=True)
        updated += 1

    print(f"\nselesai: updated={updated} skipped={skipped}", flush=True)
    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
