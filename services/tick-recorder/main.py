from contextlib import asynccontextmanager
from datetime import datetime, timezone
from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
import json
import os
import websockets
from sqlalchemy import insert
from shared.db.models import MarketSnapshot
from shared.db.session import AsyncSessionLocal, init_db, close_db

class Tick(BaseModel):
    pair: str
    mark_price: float
    index_price: float = 0
    last_price: float = 0
    bid_price: float = 0
    ask_price: float = 0
    bid_size: float = 0
    ask_size: float = 0
    funding_rate: float | None = None
    open_interest: float | None = None
    volume_24h: float | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    task = asyncio.create_task(stream_loop()) if os.getenv("TICK_RECORDER_ENABLED", "false").lower() == "true" else None
    yield
    if task:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    await close_db()

app = FastAPI(title="Tick Recorder", lifespan=lifespan)

async def save_snapshot(values: dict):
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    bid, ask = float(values.get("bid_price", 0)), float(values.get("ask_price", 0))
    values.update(timestamp=now, spread=max(ask - bid, 0), source="binance")
    async with AsyncSessionLocal() as db:
        await db.execute(insert(MarketSnapshot).values(**values)); await db.commit()

async def stream_loop():
    symbols = [s.strip().lower() for s in os.getenv("TICK_SYMBOLS", "btcusdt").split(",") if s.strip()]
    base = os.getenv("BINANCE_WS_URL", "wss://fstream.binance.com")
    streams = "/".join(f"{s}@bookTicker" for s in symbols)
    uri = f"{base.rstrip('/')}/stream?streams={streams}"
    while True:
        try:
            async with websockets.connect(uri, ping_interval=20, ping_timeout=10) as ws:
                async for raw in ws:
                    payload = json.loads(raw).get("data", raw)
                    if payload.get("e") != "bookTicker": continue
                    bid, ask = float(payload.get("b", 0)), float(payload.get("a", 0))
                    await save_snapshot({"pair": payload.get("s", ""), "mark_price": (bid + ask) / 2,
                        "index_price": (bid + ask) / 2, "last_price": (bid + ask) / 2,
                        "bid_price": bid, "ask_price": ask,
                        "bid_size": float(payload.get("B", 0)), "ask_size": float(payload.get("A", 0))})
        except asyncio.CancelledError: raise
        except Exception: await asyncio.sleep(5)

@app.get("/health")
async def health(): return {"status": "healthy", "service": "tick-recorder"}

@app.post("/ticks")
async def record(tick: Tick):
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    values = tick.model_dump()
    bid, ask = values.pop("bid_price"), values.pop("ask_price")
    values.update(timestamp=now, spread=max(ask - bid, 0), source="binance",
                  bid_price=bid, ask_price=ask)
    async with AsyncSessionLocal() as db:
        await db.execute(insert(MarketSnapshot).values(**values)); await db.commit()
    return {"status": "recorded", "pair": tick.pair.upper(), "timestamp": now.isoformat()}

if __name__ == "__main__":
    import uvicorn; uvicorn.run(app, host="0.0.0.0", port=8000)
