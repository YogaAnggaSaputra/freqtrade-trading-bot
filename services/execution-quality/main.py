from fastapi import FastAPI
from pydantic import BaseModel
from shared.quant.execution import slippage_bps, choose_order_type
from shared.quant.stats_advanced import implementation_shortfall

app = FastAPI(title="Execution Quality Engine", version="2.0.0 (Implementation Shortfall Edition)")

class Fill(BaseModel):
    reference_price: float
    fill_price: float
    decision_price: float | None = None
    arrival_price: float | None = None
    side: str = "buy"
    spread_bps: float = 0.0
    fee_bps: float = 4.0
    urgent: bool = False

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "execution-quality", "edition": "implementation-shortfall"}

@app.post("/analyze")
async def analyze(fill: Fill):
    slip = slippage_bps(fill.reference_price, fill.fill_price)
    rec_type = choose_order_type(fill.spread_bps, slip, fill.urgent)

    shortfall = None
    if fill.decision_price and fill.arrival_price:
        shortfall = implementation_shortfall(
            decision_price=fill.decision_price,
            arrival_price=fill.arrival_price,
            execution_price=fill.fill_price,
            side=fill.side,
            fee_bps=fill.fee_bps,
        )

    return {
        "slippage_bps": slip,
        "recommended_order_type": rec_type,
        "implementation_shortfall": shortfall,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
