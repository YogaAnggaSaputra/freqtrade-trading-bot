from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import select
from shared.db.models import ExitRegret
from shared.db.session import AsyncSessionLocal, close_db, init_db

@asynccontextmanager
async def lifespan(app):
    await init_db(); yield; await close_db()

app = FastAPI(title="Post Exit Regret", lifespan=lifespan)
class ExitObservation(BaseModel):
    trade_id: str; pair: str
    exit_price: float; future_prices: list[float]; side: str = "long"
@app.get("/health")
async def health(): return {"status": "healthy", "service": "post-exit-regret"}
@app.post("/analyze")
async def analyze(o: ExitObservation):
    if not o.future_prices: return {"regret": 0, "classification": "unknown"}
    best = max(o.future_prices) if o.side.lower() == "long" else min(o.future_prices)
    regret = (best - o.exit_price) / o.exit_price if o.side.lower() == "long" else (o.exit_price - best) / o.exit_price
    classification = "too_early" if regret > .01 else "acceptable"
    async with AsyncSessionLocal() as db:
        existing = (await db.execute(select(ExitRegret).where(ExitRegret.trade_id == o.trade_id))).scalar_one_or_none()
        if not existing:
            db.add(ExitRegret(trade_id=o.trade_id, pair=o.pair, exit_price=o.exit_price,
                              best_future_price=best, regret_pct=regret,
                              classification=classification, horizon_candles=len(o.future_prices)))
            await db.commit()
    return {"trade_id": o.trade_id, "regret": regret, "best_future_price": best, "classification": classification}
if __name__ == "__main__":
    import uvicorn; uvicorn.run(app, host="0.0.0.0", port=8000)
