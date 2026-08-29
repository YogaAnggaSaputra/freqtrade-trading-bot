from fastapi import FastAPI
from pydantic import BaseModel
from shared.quant.allocation import allocate
from shared.quant.portfolio import risk_parity_weights, min_variance_weights, covariance_matrix

app = FastAPI(title="Capital Allocation Engine", version="2.0.0 (Risk Parity Edition)")

class AllocationRequest(BaseModel):
    capital: float
    drawdown: float = 0.0
    regime: str = "unknown"
    volatilities: dict[str, float]
    returns_series: dict[str, list[float]] | None = None
    mode: str = "volatility_parity"  # "volatility_parity" | "risk_parity" | "min_variance"

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "capital-allocation", "edition": "risk-parity"}

@app.post("/allocate")
async def allocation(req: AllocationRequest):
    if req.returns_series and len(req.returns_series) >= 2 and req.mode in ("risk_parity", "min_variance"):
        cov = covariance_matrix(req.returns_series)
        if req.mode == "risk_parity":
            weights = risk_parity_weights(cov)
        else:
            weights = min_variance_weights(cov)
        base = allocate(req.capital, req.drawdown, req.regime, req.volatilities)
        multiplier = base.get("exposure_multiplier", 1.0)
        return {
            "exposure_multiplier": multiplier,
            "allocations": {p: req.capital * multiplier * w for p, w in weights.items()},
            "weights": weights,
            "mode": req.mode,
        }
    return allocate(req.capital, req.drawdown, req.regime, req.volatilities)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
