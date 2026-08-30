#!/usr/bin/env python3
"""Backfill mae_pct/mfe_pct (excursion nyata) ke trade_outcomes.features.

Sumber: SQLite tradesv3 kolom max_rate/min_rate/open_rate/is_short.
MAE = adverse excursion (harga terburuk vs entry).
MFE = favorable excursion (harga terbaik vs entry).
Dipakai retrainer MAE/MFE fallback dari trade_outcomes.

Anti look-ahead TIDAK relevan di sini: max_rate/min_rate memang excursion
AKTUAL sepanjang hidup trade (label target, bukan feature input). Ini label
supervised — justru harus dari realisasi trade.

Usage (dalam container freqtrade-runtime):
  python backfill_excursion.py [--dry-run]
"""
import argparse
import asyncio
import json
import os
import sqlite3

import asyncpg

SQLITE = "/freqtrade/user_data/tradesv3.sqlite"


def load_excursion() -> dict:
    """trade_id → (mae_pct, mfe_pct) dari SQLite."""
    c = sqlite3.connect(f"file:{SQLITE}?mode=ro", uri=True).cursor()
    c.execute("""
        SELECT id, open_rate, max_rate, min_rate, is_short
        FROM trades
        WHERE max_rate IS NOT NULL AND min_rate IS NOT NULL AND open_rate > 0
    """)
    out = {}
    for tid, entry, mx, mn, is_short in c.fetchall():
        entry = float(entry)
        mx = float(mx)
        mn = float(mn)
        if is_short:
            # short: adverse = harga naik (mx), favorable = harga turun (mn)
            mae = (mx - entry) / entry
            mfe = (entry - mn) / entry
        else:
            # long: adverse = harga turun (mn), favorable = harga naik (mx)
            mae = (entry - mn) / entry
            mfe = (mx - entry) / entry
        out[int(tid)] = (max(0.0, mae), max(0.0, mfe))
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    exc = load_excursion()
    print(f"excursion dari SQLite: {len(exc)} trade", flush=True)

    pool = await asyncpg.create_pool(
        host=os.getenv("DB_HOST", "postgres"),
        port=int(os.getenv("DB_PORT", "5432")),
        user=os.getenv("DB_USER", "botbinance"),
        password=os.getenv("DB_PASSWORD", "changeme"),
        database=os.getenv("DB_NAME", "botbinance"),
        min_size=1, max_size=3,
    )

    rows = await pool.fetch("""
        SELECT trade_id, entry_conditions
        FROM trade_outcomes
        WHERE entry_conditions ? 'features'
          AND (SELECT count(*) FROM jsonb_object_keys(entry_conditions->'features')) >= 5
    """)
    updated = skipped = 0
    for r in rows:
        tid = r["trade_id"]
        if tid not in exc:
            skipped += 1
            continue
        mae, mfe = exc[tid]
        raw = r["entry_conditions"]
        ec = json.loads(raw) if isinstance(raw, str) else dict(raw)
        feats = dict(ec.get("features", {}))
        feats["mae_pct"] = mae
        feats["mfe_pct"] = mfe
        ec["features"] = feats
        if args.dry_run:
            print(f"  [{tid}] mae={mae:.4f} mfe={mfe:.4f} (DRY)", flush=True)
        else:
            await pool.execute(
                "UPDATE trade_outcomes SET entry_conditions=$1 WHERE trade_id=$2",
                json.dumps(ec, default=str), tid,
            )
            print(f"  [{tid}] mae={mae:.4f} mfe={mfe:.4f} UPDATED", flush=True)
        updated += 1

    print(f"\nselesai: updated={updated} skipped={skipped}", flush=True)
    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
