"""Backfill/refresh exit regret observations from stored market candles.

Run periodically (cron/container) after the market-data gateway has persisted
the post-exit candles. It is idempotent on trade_id.
"""
from __future__ import annotations
import asyncio
import os
from datetime import UTC, datetime, timedelta
from sqlalchemy import text
from shared.db.models import ExitRegret, TradeOutcome
from shared.db.session import AsyncSessionLocal, init_db, close_db

async def run_once() -> int:
    inserted = 0
    async with AsyncSessionLocal() as db:
        outcomes = (await db.execute(text("""
          SELECT t.trade_id, t.pair, t.timestamp_exit, t.entry_conditions
          FROM trade_outcomes t LEFT JOIN exit_regrets r ON r.trade_id = CAST(t.trade_id AS TEXT)
          WHERE r.id IS NULL AND t.timestamp_exit < :ready
          ORDER BY t.timestamp_exit LIMIT 100
        """), {"ready": datetime.now(UTC).replace(tzinfo=None) - timedelta(minutes=105)})).mappings().all()
        for outcome in outcomes:
            conditions = outcome.get("entry_conditions") or {}
            side = str(conditions.get("side", "long")).lower()
            exit_price = float(conditions.get("exit_rate", 0) or 0)
            if exit_price <= 0: continue
            candles = (await db.execute(text("""
              SELECT close FROM market_candles
              WHERE pair=:pair AND timeframe='5m' AND timestamp > :exit
              ORDER BY timestamp LIMIT 20
            """), {"pair": outcome["pair"], "exit": outcome["timestamp_exit"]})).scalars().all()
            if not candles: continue
            prices = [float(v) for v in candles]
            from shared.quant.supreme_final import counterfactual_exit_regret
            cfr = counterfactual_exit_regret(prices, exit_price, side, decay_gamma=0.05)
            best = max(prices) if side != "short" else min(prices)
            regret = cfr["opportunity_loss"]
            db.add(ExitRegret(trade_id=str(outcome["trade_id"]), pair=outcome["pair"],
                              exit_price=exit_price, best_future_price=best,
                              regret_pct=regret, classification="too_early" if regret > .01 else "acceptable",
                              horizon_candles=len(prices)))
            inserted += 1
        await db.commit()
    return inserted

async def main():
    await init_db()
    try:
        interval = int(os.getenv("REGRET_WORKER_INTERVAL_SECONDS", "300"))
        while True:
            try: print(f"post-exit regret: inserted={await run_once()}", flush=True)
            except Exception as exc: print(f"post-exit regret error: {exc}", flush=True)
            await asyncio.sleep(interval)
    finally: await close_db()

if __name__ == "__main__": asyncio.run(main())
