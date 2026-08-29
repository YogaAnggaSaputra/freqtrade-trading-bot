from fastapi import FastAPI
from pydantic import BaseModel
from shared.quant.stats_advanced import mahalanobis_distance_2d, evt_pareto_tail_index

app = FastAPI(title="Anomaly Detection Engine", version="2.0.0 (EVT & Mahalanobis Edition)")

class CandleCheck(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    return_zscore: float | None = None
    volume_zscore: float | None = None
    historical_returns: list[float] | None = None
    baseline_zscores: list[tuple[float, float]] | None = None

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "anomaly-detection", "edition": "evt-mahalanobis"}

@app.post("/check")
async def check(c: CandleCheck):
    invalid = c.high < max(c.open, c.close, c.low) or c.low > min(c.open, c.close, c.high)
    z_ret = abs(c.return_zscore or 0.0)
    z_vol = abs(c.volume_zscore or 0.0)

    # 1. Mahalanobis distance (bivariate anomaly detection: return_z & vol_z)
    mahalanobis_d = 0.0
    if c.baseline_zscores:
        mahalanobis_d = mahalanobis_distance_2d((z_ret, z_vol), c.baseline_zscores)

    # 2. Extreme Value Theory (EVT) tail analysis
    evt_stats = evt_pareto_tail_index(c.historical_returns) if c.historical_returns else {"xi": 0.0}

    reasons = (["invalid_ohlc"] if invalid else []) + \
              (["return_gt_4_sigma"] if z_ret > 4 else []) + \
              (["mahalanobis_anomaly"] if mahalanobis_d > 3.5 else []) + \
              (["heavy_tail_crash_risk"] if evt_stats.get("xi", 0) > 0.40 else [])

    severity = "critical" if invalid or mahalanobis_d > 4.5 or z_ret > 5 else ("high" if reasons else "normal")

    return {
        "anomalous": bool(reasons),
        "reasons": reasons,
        "severity": severity,
        "mahalanobis_distance": round(mahalanobis_d, 2),
        "evt_tail_index_xi": round(evt_stats.get("xi", 0.0), 3),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
