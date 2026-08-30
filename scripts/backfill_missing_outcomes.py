#!/usr/bin/env python3
"""Tarik trade closed dari SQLite yang BELUM ada di trade_outcomes (Postgres).

Bikin ROW BARU di trade_outcomes untuk tiap trade SQLite yg missing, lengkap:
- entry_conditions.features  → rekonstruksi dari market_candles (candle <= open_date, anti look-ahead)
- entry_conditions.mae_pct/mfe_pct → excursion nyata dari max_rate/min_rate (label)
- pnl_pct/pnl_abs, predicted_rr(actual), regime, timestamps

Reuse compute_features dari backfill_features (identik indikator strategy).

Usage (dalam container freqtrade-runtime, /tmp):
  python backfill_missing_outcomes.py [--dry-run]
"""
import argparse
import asyncio
import importlib.util
import json
import os
import sqlite3
from datetime import timezone

import asyncpg
import pandas as pd

SQLITE = "/freqtrade/user_data/tradesv3.sqlite"
MIN_CANDLES = 60

# import compute_features dari backfill_features.py (harus di /tmp yg sama)
_spec = importlib.util.spec_from_file_location("bf", "/tmp/backfill_features.py")
_bf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bf)
compute_features = _bf.compute_features


def load_sqlite_trades() -> list[dict]:
    conn = sqlite3.connect(f"file:{SQLITE}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT id, pair, open_rate, close_rate, max_rate, min_rate, is_short,
               close_profit, close_profit_abs, open_date, close_date,
               exit_reason, timeframe, stop_loss_pct
        FROM trades WHERE is_open=0 AND close_rate IS NOT NULL
    """).fetchall()
    return [dict(r) for r in rows]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pool = await asyncpg.create_pool(
        host=os.getenv("DB_HOST", "postgres"),
        port=int(os.getenv("DB_PORT", "5432")),
        user=os.getenv("DB_USER", "botbinance"),
        password=os.getenv("DB_PASSWORD", "changeme"),
        database=os.getenv("DB_NAME", "botbinance"),
        min_size=1, max_size=3,
    )

    existing = {r["trade_id"] for r in await pool.fetch("SELECT trade_id FROM trade_outcomes")}
    trades = load_sqlite_trades()
    missing = [t for t in trades if t["id"] not in existing]
    print(f"SQLite closed: {len(trades)} | sudah ada: {len(existing)} | missing: {len(missing)}", flush=True)

    inserted = skipped = 0
    for t in missing:
        tid = t["id"]
        pair = t["pair"]
        entry = float(t["open_rate"])
        is_short = bool(t["is_short"])

        # parse timestamps (SQLite simpan naive UTC string)
        ts_entry = pd.to_datetime(t["open_date"]).to_pydatetime()
        ts_exit = pd.to_datetime(t["close_date"]).to_pydatetime()
        ts_entry_naive = ts_entry.replace(tzinfo=None)

        # candle <= entry (anti look-ahead untuk FITUR input)
        candles = await pool.fetch("""
            SELECT open, high, low, close, volume
            FROM market_candles
            WHERE pair=$1 AND timeframe='5m' AND timestamp <= $2
            ORDER BY timestamp DESC LIMIT 400
        """, pair, ts_entry_naive)
        if len(candles) < MIN_CANDLES:
            skipped += 1
            print(f"  [{tid}] {pair}: SKIP (candle {len(candles)}<{MIN_CANDLES})", flush=True)
            continue

        df = pd.DataFrame([dict(c) for c in reversed(candles)],
                          columns=["open", "high", "low", "close", "volume"]).astype(float)
        feats = compute_features(df)
        if len(feats) < 5:
            skipped += 1
            print(f"  [{tid}] {pair}: SKIP (fitur {len(feats)}<5)", flush=True)
            continue

        # excursion nyata (label MAE/MFE)
        mx = float(t["max_rate"]) if t["max_rate"] else entry
        mn = float(t["min_rate"]) if t["min_rate"] else entry
        if is_short:
            mae = max(0.0, (mx - entry) / entry)
            mfe = max(0.0, (entry - mn) / entry)
        else:
            mae = max(0.0, (entry - mn) / entry)
            mfe = max(0.0, (mx - entry) / entry)
        feats["mae_pct"] = mae
        feats["mfe_pct"] = mfe
        feats["feature_version"] = "v1"
        feats["candle_count"] = len(df)

        pnl_pct = float(t["close_profit"] or 0.0)
        pnl_abs = float(t["close_profit_abs"] or 0.0)
        # actual RR proxy: pnl / risk(stop_loss_pct)
        sl_pct = abs(float(t["stop_loss_pct"] or 0.0)) or None
        actual_rr = (pnl_pct / sl_pct) if sl_pct else None

        # Sanitize Infinity/NaN → None sebelum JSON encode (Postgres JSONB
        # reject literal "Infinity"/"NaN" token — error: invalid input syntax
        # for type json, Token "Infinity" is invalid)
        def _safe(v):
            import math
            if v is None:
                return None
            if isinstance(v, float):
                if math.isnan(v) or math.isinf(v):
                    return None
                return v
            if isinstance(v, dict):
                return {k: _safe(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [_safe(x) for x in v]
            return v

        ec = {
            "regime": feats.get("regime", "RANGING"),
            "side": "short" if is_short else "long",
            "entry_rate": entry,
            "exit_rate": float(t["close_rate"]),
            "sl_pct": sl_pct,
            "features": _safe(feats),
            "source": "sqlite_backfill",
        }

        if args.dry_run:
            print(f"  [{tid}] {pair}: {len(feats)} fitur, mae={mae:.4f} mfe={mfe:.4f} pnl={pnl_pct:.4f} (DRY)", flush=True)
        else:
            await pool.execute("""
                INSERT INTO trade_outcomes
                  (trade_id, pair, timeframe, entry_conditions, exit_reason,
                   pnl_pct, pnl_abs, predicted_rr, actual_rr, regime_at_entry,
                   timestamp_entry, timestamp_exit)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
                ON CONFLICT (trade_id) DO NOTHING
            """, tid, pair, str(t["timeframe"] or "5m"), json.dumps(ec, default=str),
                t["exit_reason"] or "unknown", pnl_pct, pnl_abs,
                None, actual_rr, feats.get("regime", "RANGING"),
                ts_entry.replace(tzinfo=timezone.utc), ts_exit.replace(tzinfo=timezone.utc))
            print(f"  [{tid}] {pair}: INSERT {len(feats)} fitur, mae={mae:.4f} mfe={mfe:.4f}", flush=True)
        inserted += 1

    print(f"\nselesai: inserted={inserted} skipped={skipped}", flush=True)
    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
