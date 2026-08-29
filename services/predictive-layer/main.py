from fastapi import FastAPI
from pydantic import BaseModel
from shared.quant.predictive import measured_move, liquidity_levels, mtf_alignment

app = FastAPI(title="Predictive Layer Engine", version="2.0.0 (Liquidity Density Edition)")

class TargetRequest(BaseModel):
    range_high: float
    range_low: float
    breakout_up: bool = True

class LiquidityRequest(BaseModel):
    highs: list[float] = []
    lows: list[float] = []
    tolerance: float = 0.001

class MomentumRequest(BaseModel):
    scores: dict[str, float]

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "predictive-layer", "edition": "kernel-liquidity"}

@app.post("/target")
async def target(r: TargetRequest):
    return {"measured_move": measured_move(r.range_high, r.range_low, r.breakout_up)}

@app.post("/liquidity-levels")
async def levels(r: LiquidityRequest):
    return {"levels": liquidity_levels(r.highs, r.lows, r.tolerance)}

@app.post("/mtf-alignment")
async def alignment(r: MomentumRequest):
    return {"alignment": mtf_alignment(r.scores), "scores": r.scores}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
