from fastapi import FastAPI
from pydantic import BaseModel, Field
from shared.quant.execution import almgren_chriss_impact

app = FastAPI(title="Execution Guard Engine", version="2.0.0 (Almgren-Chriss Edition)")

class OrderCheck(BaseModel):
    pair: str
    price: float = Field(gt=0)
    mid_price: float = Field(gt=0)
    amount: float = Field(gt=0)
    leverage: int = Field(default=1, ge=1)
    min_notional: float = Field(default=5, gt=0)
    adv_24h_usdt: float = Field(default=1000000.0, gt=0)
    volatility: float = Field(default=0.02, gt=0)
    pending_same_order: bool = False

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "execution-guard", "edition": "almgren-chriss"}

@app.post("/validate")
async def validate(order: OrderCheck):
    reasons = []
    if order.pending_same_order:
        reasons.append("duplicate_pending_order")

    notional = order.amount * order.price * order.leverage
    if notional < order.min_notional:
        reasons.append("min_notional_not_met")

    if abs(order.price - order.mid_price) / order.mid_price > 0.05:
        reasons.append("price_deviation_gt_5pct")

    # Almgren-Chriss Square-Root Market Impact Check
    impact = almgren_chriss_impact(
        order_size=notional,
        adv=order.adv_24h_usdt,
        volatility=order.volatility,
    )

    if not impact["is_safe_notional"]:
        reasons.append(f"market_impact_excessive:{impact['total_impact_bps']}bps_gt_25bps")

    return {
        "approved": not reasons,
        "reasons": reasons,
        "pair": order.pair.upper(),
        "notional_usdt": round(notional, 2),
        "almgren_chriss_impact": impact,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
