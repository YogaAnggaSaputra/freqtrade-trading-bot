from fastapi import FastAPI
from pydantic import BaseModel
from shared.quant.orderbook import dom_pressure, spoofing_score, vpin_like, iceberg_evidence
from shared.quant.microstructure import vpin_toxicity, kyle_lambda

app = FastAPI(title="Orderbook Intelligence Engine", version="2.0.0 (Toxicity Edition)")

class Book(BaseModel):
    bid_volume: float
    ask_volume: float
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    previous_level_size: float = 0.0
    current_level_size: float = 0.0
    traded_size: float = 0.0
    refill_count: int = 0
    displayed_size: float = 0.0
    executed_size: float = 0.0
    # Advanced Microstructure fields
    volume_buckets: list[tuple[float, float]] | None = None
    price_changes: list[float] | None = None
    net_volumes: list[float] | None = None

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "orderbook-intelligence", "edition": "vpin-toxicity"}

@app.post("/analyze")
async def analyze(b: Book):
    pressure = dom_pressure(b.bid_volume, b.ask_volume)
    vpin_val = vpin_toxicity(b.volume_buckets) if b.volume_buckets else vpin_like(b.buy_volume, b.sell_volume)
    lambda_val = kyle_lambda(b.price_changes, b.net_volumes) if b.price_changes and b.net_volumes else 0.0

    return {
        "dom_pressure": pressure,
        "vpin_toxicity": vpin_val,
        "kyle_lambda": lambda_val,
        "spoofing_score": spoofing_score(b.previous_level_size, b.current_level_size, b.traded_size),
        "iceberg_evidence": iceberg_evidence(b.refill_count, b.displayed_size, b.executed_size),
        "toxic_flow_alert": vpin_val > 0.50,
        "signal": "buy" if pressure > 0.15 and vpin_val <= 0.50 else ("sell" if pressure < -0.15 and vpin_val <= 0.50 else "neutral"),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
