import os
import math
import time
import aiohttp
from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import text
from shared.db.session import AsyncSessionLocal
from shared.quant.microstructure import amihud_illiquidity

app = FastAPI(title="Adaptive Whitelist Engine", version="2.0.0 (Amihud Liquidity Edition)")
_whitelist_cache: tuple[float, dict] | None = None

class PairStat(BaseModel):
    pair: str
    volume_24h: float = 0.0
    atr_pct: float = 0.0
    score: float = 0.0
    amihud: float = 0.0

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "adaptive-whitelist", "edition": "amihud-liquidity"}

@app.post("/rank")
async def rank(pairs: list[PairStat], limit: int = 50):
    """Rank pairs: Amihud-penalized liquidity score × sqrt(volume) × (1 + edge).

    Pairs with high Amihud illiquidity ratio get penalised — avoids entering
    into thinly-traded markets where market impact would eat the edge.
    """
    eligible = [p for p in pairs if 0.005 <= p.atr_pct <= 0.05]
    def _composite(p: PairStat) -> float:
        amihud_penalty = 1.0 / (1.0 + max(0.0, p.amihud))
        return p.score * amihud_penalty
    ranked = sorted(eligible, key=_composite, reverse=True)
    return {"pairs": [p.pair for p in ranked[:limit]], "count": len(eligible)}

@app.get("/whitelist")
async def whitelist(limit: int = 50, refresh: bool = False):
    """Build a live whitelist from Binance 24h data with Amihud-penalized ranking."""
    global _whitelist_cache
    if _whitelist_cache and not refresh and time.time() - _whitelist_cache[0] < 3600:
        cached = dict(_whitelist_cache[1])
        cached["cached"] = True
        return cached
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
            async with session.get("https://fapi.binance.com/fapi/v1/ticker/24hr") as response:
                rows = await response.json()

        historical_edge: dict[str, float] = {}
        try:
            regime = os.getenv("WHITELIST_REGIME", "")
            async with AsyncSessionLocal() as db:
                history = (await db.execute(text("""
                    SELECT pair, AVG(pnl_pct) AS edge, COUNT(*) AS samples
                    FROM trade_outcomes
                    WHERE (:regime = '' OR COALESCE(regime_at_entry, '') = :regime)
                    GROUP BY pair HAVING COUNT(*) >= 5
                """), {"regime": regime})).mappings().all()
            historical_edge = {
                str(row["pair"]).split("/")[0].upper(): float(row["edge"] or 0)
                for row in history
            }
        except Exception:
            historical_edge = {}

        pairs = []
        for row in rows:
            symbol = str(row.get("symbol", ""))
            if not symbol.endswith("USDT") or symbol.endswith(("USDCUSDT", "BUSDUSDT")):
                continue
            low = float(row.get("lowPrice", 0) or 0)
            high = float(row.get("highPrice", 0) or 0)
            atr_proxy = (high - low) / low if low else 0
            if not (0.005 <= atr_proxy <= 0.05):
                continue
            base = symbol[:-4]
            volume = float(row.get("quoteVolume", 0) or 0)
            price = float(row.get("lastPrice", 0) or 0)
            edge = max(-1.0, min(1.0, historical_edge.get(base, 0.0) * 100.0))
            # Amihud illiquidity proxy: (atr_proxy as abs_return) / (volume_24h)
            amihud = amihud_illiquidity([atr_proxy], [max(volume, 1.0)]) if volume > 0 else 999.0
            base_score = math.sqrt(max(volume, 1.0)) * (1.0 + edge * 0.10)
            pairs.append(PairStat(pair=f"{base}/USDT:USDT", volume_24h=volume,
                                  atr_pct=atr_proxy, score=base_score, amihud=amihud))

        result = {**(await rank(pairs, limit)), "source": "binance_24h", "refresh_seconds": 3600, "cached": False}
        _whitelist_cache = (time.time(), result)
        return result
    except Exception as exc:
        return {"pairs": [], "count": 0, "source": "fallback", "error": str(exc)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
