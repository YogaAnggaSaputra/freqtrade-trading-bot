"""Build UTC hour/day regime and outcome calendar from stored outcomes."""
from __future__ import annotations
import asyncio
from sqlalchemy import text
from shared.db.session import AsyncSessionLocal

async def main():
    async with AsyncSessionLocal() as db:
        rows = (await db.execute(text("""
          SELECT EXTRACT(HOUR FROM timestamp_entry)::int AS utc_hour,
                 EXTRACT(ISODOW FROM timestamp_entry)::int AS iso_day,
                 COALESCE(regime_at_entry,'unknown') AS regime,
                 COUNT(*) AS trades, AVG(pnl_pct) AS avg_pnl,
                 AVG(CASE WHEN pnl_pct > 0 THEN 1.0 ELSE 0.0 END) AS win_rate
          FROM trade_outcomes GROUP BY utc_hour, iso_day, regime
          ORDER BY iso_day, utc_hour, regime
        """))).mappings().all()
    for row in rows:
        print(f"day={row['iso_day']} hour={row['utc_hour']:02} regime={row['regime']:20} trades={row['trades']:4} win={float(row['win_rate']):.1%} avg={float(row['avg_pnl']):.5f}")

if __name__ == "__main__": asyncio.run(main())
