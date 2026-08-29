"""Generate compact P&L attribution by regime from trade_outcomes."""
import asyncio
from sqlalchemy import text
from shared.db.session import AsyncSessionLocal

async def main():
    async with AsyncSessionLocal() as db:
        rows = (await db.execute(text("""
            SELECT COALESCE(regime_at_entry, 'unknown') regime,
                   COUNT(*) trades, COALESCE(SUM(pnl_abs), 0) pnl,
                   AVG(CASE WHEN pnl_abs > 0 THEN 1.0 ELSE 0.0 END) win_rate
            FROM trade_outcomes GROUP BY regime_at_entry ORDER BY pnl DESC
        """))).mappings().all()
    for row in rows:
        print(f"{row['regime']:24} trades={row['trades']:5} win={float(row['win_rate']):.1%} pnl={float(row['pnl']):.4f}")

if __name__ == "__main__":
    asyncio.run(main())
