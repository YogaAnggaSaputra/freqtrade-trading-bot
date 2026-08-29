from contextlib import asynccontextmanager
from datetime import datetime, UTC

from fastapi import FastAPI
from pydantic import BaseModel
from shared.db.models import PositionHealthSnapshot
from shared.db.session import AsyncSessionLocal, close_db, init_db
from shared.quant.position_risk import funding_impact, stress_loss, kill_switch_level
from shared.quant.position import position_health

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield
    await close_db()

app = FastAPI(title="Position Monitor", lifespan=lifespan)
from shared.quant.supreme_final import chandelier_exit_ratchet

class RiskRequest(BaseModel):
    notional: float; funding_rate: float = 0; funding_periods: int = 0
    expected_profit: float = 0; adverse_move_pct: float = .05; correlation_impact: float = 0
    daily_drawdown: float = 0; latency_multiplier: float = 1; anomaly: bool = False
    trade_id: str | None = None; pair: str | None = None; regime: str = "unknown"
    regime_at_entry: str | None = None; current_r: float = 0; peak_r: float = 0
    mtf_alignment: float = 1.0; highest_price: float = 0.0; current_atr: float = 0.0
    previous_stop: float = 0.0; side: str = "long"
@app.get("/health")
async def health(): return {"status": "healthy", "service": "position-monitor"}
@app.post("/assess")
async def assess(req: RiskRequest):
    funding = funding_impact(req.notional, req.funding_rate, req.funding_periods, req.expected_profit)
    stress = stress_loss(req.notional, req.adverse_move_pct, req.correlation_impact)
    health_snapshot = position_health(
        req.current_r, req.peak_r, req.regime, req.regime_at_entry, req.mtf_alignment,
    )
    chandelier_stop = chandelier_exit_ratchet(
        highest_price=req.highest_price or req.notional,
        current_atr=req.current_atr or (req.notional * 0.01),
        previous_stop=req.previous_stop,
        atr_multiplier=3.0,
        side=req.side,
    ) if req.highest_price > 0 else req.previous_stop
    persisted = False
    if req.trade_id and req.pair:
        async with AsyncSessionLocal() as db:
            db.add(PositionHealthSnapshot(
                trade_id=req.trade_id,
                pair=req.pair,
                timestamp=datetime.now(UTC).replace(tzinfo=None),
                health_score=health_snapshot["score"],
                thesis_valid=health_snapshot["thesis_valid"],
                momentum_decay=health_snapshot["momentum_decay"],
                regime=req.regime,
                details={"funding": funding, "stress": stress},
            ))
            await db.commit()
            persisted = True
    return {"funding": funding, "stress": stress,
            "position_health": health_snapshot, "chandelier_stop": chandelier_stop, "persisted": persisted,
            "kill_switch_level": kill_switch_level(req.daily_drawdown, req.latency_multiplier, req.anomaly)}
if __name__ == "__main__":
    import uvicorn; uvicorn.run(app, host="0.0.0.0", port=8000)
