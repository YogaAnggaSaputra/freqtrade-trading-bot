import time
import os
import aiohttp
from fastapi import FastAPI
from shared.quant.onchain_offchain import whale_netflow_score, defiliquidation_cascade_risk

app = FastAPI(title="On-Chain Engine", version="2.0.0 (Whale Netflow & Liquidation Cascade Edition)")
_cache: dict[str, tuple[float, dict]] = {}
_open_interest_history: dict[str, tuple[float, float]] = {}

async def _fetch(symbol: str) -> dict:
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=4)) as session:
        result = {"pair": symbol, "funding_rates": {}, "open_interest": None,
                  "open_interest_delta_pct": 0.0, "funding_extreme": False,
                  "netflow": None, "netflow_signal": None, "large_transactions": [],
                  "data_quality": [], "source": "fallback"}
        try:
            async with session.get("https://fapi.binance.com/fapi/v1/premiumIndex", params={"symbol": symbol}) as response:
                data = await response.json()
            result["funding_rates"]["binance"] = float(data.get("lastFundingRate", 0))
            result["data_quality"].append("binance_funding")
            result["source"] = "binance"
        except Exception: pass
        try:
            async with session.get("https://fapi.binance.com/fapi/v1/openInterest", params={"symbol": symbol}) as response:
                data = await response.json()
            result["open_interest"] = float(data.get("openInterest", 0))
            previous = _open_interest_history.get(symbol)
            if previous and previous[1] > 0:
                result["open_interest_delta_pct"] = (result["open_interest"] - previous[1]) / previous[1]
            else:
                result["open_interest_delta_pct"] = 0.0
            _open_interest_history[symbol] = (time.time(), result["open_interest"])
            result["data_quality"].append("binance_open_interest")
        except Exception:
            pass
        try:
            async with session.get("https://api.bybit.com/v5/market/tickers", params={"category": "linear", "symbol": symbol}) as response:
                data = await response.json()
            row = (data.get("result", {}).get("list") or [{}])[0]
            result["funding_rates"]["bybit"] = float(row.get("fundingRate", 0))
            result["data_quality"].append("bybit_funding")
        except Exception: pass
        try:
            okx_symbol = symbol.replace("USDT", "-USDT-SWAP")
            async with session.get("https://www.okx.com/api/v5/public/funding-rate", params={"instId": okx_symbol}) as response:
                data = await response.json()
            row = (data.get("data") or [{}])[0]
            result["funding_rates"]["okx"] = float(row.get("fundingRate", 0))
            result["data_quality"].append("okx_funding")
        except Exception: pass
        # Optional normalized on-chain provider. The engine intentionally does
        # not invent netflow/whale data when no provider is configured.
        provider_url = os.getenv("ONCHAIN_DATA_PROVIDER_URL", "").strip()
        if provider_url:
            try:
                headers = {}
                api_key = os.getenv("ONCHAIN_DATA_PROVIDER_API_KEY", "").strip()
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                async with session.get(provider_url, params={"symbol": symbol}, headers=headers) as response:
                    payload = await response.json(content_type=None)
                if isinstance(payload, dict):
                    inflow = float(payload.get("inflow_volume", payload.get("netflow", 0.0) if float(payload.get("netflow", 0.0)) > 0 else 0.0))
                    outflow = float(payload.get("outflow_volume", abs(float(payload.get("netflow", 0.0))) if float(payload.get("netflow", 0.0)) < 0 else 0.0))
                    netflow_calc = whale_netflow_score(inflow, outflow, float(payload.get("average_volume", 1.0)))
                    result["netflow"] = netflow_calc["netflow_amount"]
                    result["netflow_signal"] = netflow_calc["netflow_score"]
                    result["whale_signal"] = netflow_calc["signal"]
                    alerts = payload.get("large_transactions", payload.get("whale_alerts", []))
                    if isinstance(alerts, list):
                        result["large_transactions"] = alerts[:100]
                    positions = payload.get("defi_positions", [])
                    if isinstance(positions, list) and positions:
                        price = float(payload.get("current_price", 0.0))
                        result["defi_liquidation_cascade"] = defiliquidation_cascade_risk(price, positions)
                    if result["netflow"] is not None or result["large_transactions"]:
                        result["data_quality"].append("external_onchain_provider")
                        result["source"] = f"{result['source']}+onchain_provider"
            except Exception:
                result["data_quality"].append("external_onchain_provider_unavailable")
        rates = list(result["funding_rates"].values())
        result["aggregated_funding_rate"] = sum(rates) / len(rates) if rates else 0.0
        result["funding_extreme"] = bool(rates) and abs(result["aggregated_funding_rate"]) >= .001
        return result

@app.get("/health")
async def health(): return {"status": "healthy", "service": "on-chain-engine"}

@app.get("/metrics/{pair}")
async def metrics(pair: str):
    symbol = pair.upper().split("/")[0].split(":")[0]
    if symbol.endswith("USDT") is False: symbol += "USDT"
    cached = _cache.get(symbol)
    if cached and time.time() - cached[0] < 30: return cached[1]
    try: result = await _fetch(symbol)
    except Exception: result = {"pair": symbol, "source": "fallback", "funding_rates": {}, "aggregated_funding_rate": 0, "open_interest": None, "open_interest_delta_pct": 0.0, "funding_extreme": False, "netflow": None, "netflow_signal": None, "large_transactions": [], "data_quality": []}
    _cache[symbol] = (time.time(), result)
    return result

if __name__ == "__main__":
    import uvicorn; uvicorn.run(app, host="0.0.0.0", port=8000)
