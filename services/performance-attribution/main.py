from contextlib import asynccontextmanager
from fastapi import FastAPI
from sqlalchemy import text
from shared.db.session import AsyncSessionLocal, init_db, close_db

@asynccontextmanager
async def lifespan(app: FastAPI): await init_db(); yield; await close_db()
app = FastAPI(title="Performance Attribution", lifespan=lifespan)
@app.get("/health")
async def health(): return {"status": "healthy", "service": "performance-attribution"}
@app.get("/report")
async def report():
    async with AsyncSessionLocal() as db:
        result = await db.execute(text("""
          SELECT COALESCE(regime_at_entry,'unknown') regime,
                 pair, COALESCE(exit_reason,'unknown') exit_reason,
                 COUNT(*) trades, COALESCE(SUM(pnl_abs),0) pnl,
                 AVG(pnl_pct) avg_pnl,
                 AVG(NULLIF(entry_conditions->>'conf_score', '')::numeric) avg_confidence,
                 AVG(NULLIF(entry_conditions->>'atr_ratio', '')::numeric) avg_atr_ratio,
                 AVG(NULLIF(predicted_rr, 0)) avg_predicted_rr,
                 AVG(NULLIF(actual_rr, 0)) avg_actual_rr
          FROM trade_outcomes GROUP BY regime_at_entry,pair,exit_reason ORDER BY pnl DESC
        """))
        rows = [dict(r) for r in result.mappings().all()]
    return {"rows": rows, "count": len(rows)}

@app.get("/exit-quality")
async def exit_quality():
    """Compare exit reasons with observed post-exit regret."""
    async with AsyncSessionLocal() as db:
        result = await db.execute(text("""
          SELECT COALESCE(t.exit_reason, 'unknown') AS exit_reason,
                 COUNT(*) AS trades, AVG(t.pnl_pct) AS avg_pnl,
                 AVG(r.regret_pct) AS avg_regret,
                 AVG(CASE WHEN r.classification = 'too_early' THEN 1.0 ELSE 0.0 END) AS too_early_rate
          FROM trade_outcomes t
          LEFT JOIN exit_regrets r ON r.trade_id = CAST(t.trade_id AS TEXT)
          GROUP BY t.exit_reason ORDER BY avg_regret DESC NULLS LAST
        """))
        rows = [dict(r) for r in result.mappings().all()]
    return {"rows": rows, "count": len(rows)}
if __name__ == "__main__":
    import uvicorn; uvicorn.run(app, host="0.0.0.0", port=8000)
