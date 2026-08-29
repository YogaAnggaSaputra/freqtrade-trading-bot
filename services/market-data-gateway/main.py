import asyncio
import json
import math
import os
from statistics import mean, pstdev
from datetime import datetime
from decimal import Decimal
from typing import Any

import pandas as pd
import structlog
import uvicorn
from fastapi import FastAPI
from liquidation_monitor import LiquidationMonitor
from orderbook_monitor import OrderBookMonitor

from shared.db.models import MarketCandle, MarketSnapshot
from shared.db.session import AsyncSessionLocal, close_db, init_db
from shared.messaging import Channels, MessageBus
from shared.security import load_secrets_into_env
from shared.metrics import add_metrics_endpoint

load_secrets_into_env()
logger = structlog.get_logger()

message_bus = MessageBus()
redis_client = None

BINANCE_FUTURES_TESTNET = os.getenv("BINANCE_FUTURES_TESTNET", "false").lower() == "true"

def _normalize_pair(symbol: str) -> str:
    """Convert SPOT format (BTCUSDT) to FUTURES format (BTC/USDT:USDT)."""
    symbol = symbol.upper().strip()
    # If already in futures format, return as is
    if "/" in symbol and ":" in symbol:
        return symbol
    # If ends with USDT, convert to futures format
    if symbol.endswith("USDT"):
        return symbol.replace("USDT", "/USDT:USDT")
    # If doesn't end with USDT (e.g., 1000RATS), assume USDT quote
    return symbol + "/USDT:USDT"

SYMBOLS = [s.strip().upper() for s in os.getenv("MARKET_SYMBOLS", "BTCUSDT").split(",") if s.strip()]
ALL_SYMBOLS = [
    _normalize_pair(s.strip())
    for s in os.getenv("MARKET_SYMBOLS_ALL", os.getenv("MARKET_SYMBOLS", "BTCUSDT")).split(",")
    if s.strip()
]
TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]
PARQUET_DIR = "/data/parquet"
WEBSOCKET_BATCH_SIZE = int(os.getenv("WEBSOCKET_BATCH_SIZE", "100"))


def _split_batches(symbols: list[str], batch_size: int) -> list[list[str]]:
    return [symbols[i : i + batch_size] for i in range(0, len(symbols), batch_size)]


async def get_candles_async(pair: str, timeframe: str, limit: int = 10):
    import asyncpg
    
    pool = await asyncpg.create_pool(
        host=os.getenv("DB_HOST", "postgres"),
        database=os.getenv("DB_NAME", "botbinance"),
        user=os.getenv("DB_USER", "botbinance"),
        password=os.getenv("DB_PASSWORD", "changeme"),
        port=int(os.getenv("DB_PORT", "5432")),
    )
    
    try:
        normalized = _normalize_pair(pair)
        sql = """
            SELECT timestamp, open, high, low, close, volume
            FROM market_candles
            WHERE pair = $1 AND timeframe = $2
            ORDER BY timestamp DESC
            LIMIT $3
        """
        rows = await pool.fetch(sql, normalized, timeframe, limit)
        return [
            {
                "timestamp": str(row["timestamp"]),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }
            for row in rows
        ]
    finally:
        await pool.close()


class MarketDataGateway:
    def __init__(self):
        self.ws_connections: dict[str, Any] = {}
        self._running = False
        self._candle_buffers: dict[str, list] = {}
        self._return_history: dict[str, list[float]] = {}
        self._last_candle_close: dict[str, float] = {}
        self._last_snapshots: dict[str, dict] = {}
        self.obi_monitor = OrderBookMonitor(
            pairs=ALL_SYMBOLS,
            testnet=BINANCE_FUTURES_TESTNET,
            message_bus=message_bus,
        )
        self.liq_monitor = LiquidationMonitor(
            pairs=ALL_SYMBOLS,
            testnet=BINANCE_FUTURES_TESTNET,
            message_bus=message_bus,
        )

    async def start(self):
        try:
            self._running = True
            await message_bus.connect()
            await init_db()
            logger.info("Starting Market Data Gateway", symbols_count=len(ALL_SYMBOLS))
            # These monitors are independent background loops.  Starting the
            # gateway without them silently disabled the order-book and
            # liquidation signals advertised by this service.
            await self.obi_monitor.start()
            await self.liq_monitor.start()
            # REST polling (fallback for blocked WebSocket)
            asyncio.create_task(self._rest_polling_loop())
            asyncio.create_task(self._persist_candles_loop())
            asyncio.create_task(self._persist_snapshots_loop())
            asyncio.create_task(self._seed_candles_loop())
            logger.info("start() completed successfully (REST polling mode)")
        except Exception as e:
            logger.error(f"start() failed: {e}", exc_info=True)
            raise

    async def _connect_ws_batch(self, batch_symbols: list[str], batch_index: int):
        """Legacy WS batch — disabled, kept for reference."""
        pass

    async def _rest_polling_loop(self):
        """REST polling dengan rate limit tracking (pengganti WebSocket yang diblok)."""
        import aiohttp
        POLL_INTERVAL = int(os.getenv("REST_POLL_INTERVAL", "30"))  # seconds — slower to avoid IP ban
        MAX_WEIGHT_PER_MIN = 1500  # safety margin under 2400 limit
        self._api_weight = 0
        self._weight_reset_at = datetime.utcnow()
        # Pre-compute watchlist set for O(1) lookup
        # Normalize: BTC/USDT:USDT → BTCUSDT, or BTCUSDT → BTCUSDT
        _watchlist = set()
        for s in ALL_SYMBOLS:
            s = s.upper().strip()
            if "/" in s:
                base = s.split("/")[0]  # BTC
                _watchlist.add(f"{base}USDT")
            else:
                _watchlist.add(s.replace(":", ""))
        logger.info(f"REST polling loop started (interval={POLL_INTERVAL}s, watching {len(_watchlist)} symbols)")
        while self._running:
            try:
                # Reset weight counter every minute
                now = datetime.utcnow()
                if (now - self._weight_reset_at).total_seconds() >= 60:
                    self._api_weight = 0
                    self._weight_reset_at = now
                # Check if we have weight budget
                if self._api_weight > MAX_WEIGHT_PER_MIN:
                    logger.warning(f"Rate limit approach: weight={self._api_weight}/{MAX_WEIGHT_PER_MIN}, sleeping 30s")
                    await asyncio.sleep(30)
                    continue
                # Fetch all tickers in 1 call (weight ~40)
                async with aiohttp.ClientSession() as session:
                    url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        if resp.status == 429 or resp.status == 418:
                            retry_after = int(resp.headers.get("Retry-After", "60"))
                            logger.error(f"Binance rate limited! HTTP {resp.status}, retry after {retry_after}s")
                            await asyncio.sleep(retry_after)
                            continue
                        if resp.status != 200:
                            logger.warning(f"REST ticker HTTP {resp.status}")
                            await asyncio.sleep(POLL_INTERVAL)
                            continue
                        # Track weight
                        weight = int(resp.headers.get("X-MBX-USED-WEIGHT-1M", "0"))
                        if weight:
                            self._api_weight = weight
                        data = await resp.json()
                        # Process each ticker → update snapshot + freshness
                        # NO watchlist filter — update ALL tickers from Binance
                        touched = 0
                        for item in data:
                            symbol = item.get("symbol", "")
                            if not symbol:
                                continue
                            normalized = symbol.upper().replace("/", "").replace(":", "")
                            # Update snapshot
                            self._last_snapshots[normalized] = {
                                "pair": _normalize_pair(symbol),
                                "timestamp": datetime.utcnow(),
                                "last_price": float(item.get("lastPrice", 0)),
                                "volume_24h": float(item.get("quoteVolume", 0)),
                                "mark_price": float(item.get("lastPrice", 0)),
                                "index_price": float(item.get("lastPrice", 0)),
                                "bid_price": float(item.get("bidPrice", 0)),
                                "ask_price": float(item.get("askPrice", 0)),
                                "bid_size": 0, "ask_size": 0,
                                "spread": float(item.get("askPrice", 0)) - float(item.get("bidPrice", 0)),
                                "funding_rate": 0, "open_interest": 0,
                                "source": "binance_rest",
                            }
                            # Touch Redis freshness
                            await self._touch_market_freshness(normalized)
                            touched += 1
                        logger.info(f"REST poll: {len(data)} tickers, {touched} watched, weight={self._api_weight}")
                await asyncio.sleep(POLL_INTERVAL)
            except asyncio.TimeoutError:
                logger.warning("REST poll timeout, retrying...")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"REST poll error: {e}")
                await asyncio.sleep(10)
                logger.info(f"WebSocket batch {batch_index} connected")
                async for message in ws:
                    if not self._running:
                        break
                    data = json.loads(message)
                    await self._handle_message(data)
            except Exception as e:
                logger.error(f"WebSocket batch {batch_index} error", error=str(e))
                if self._running:
                    await asyncio.sleep(5)

    async def _handle_message(self, data: dict):
        if "stream" not in data or "data" not in data:
            return
        stream = data["stream"]
        message_data = data["data"]
        if "@kline_" in stream:
            tf = stream.split("@")[1].replace("kline_", "")
            await self._handle_candle(message_data, tf)
        elif "@ticker" in stream:
            await self._handle_ticker(stream.split("@")[0], message_data)

    async def _handle_ticker(self, symbol: str, item: dict):
        normalized_pair = _normalize_pair(symbol)
        last_price = float(item.get("c", item.get("lastPrice", 0)) or 0)
        bid_price = float(item.get("b", item.get("bidPrice", 0)) or 0)
        ask_price = float(item.get("a", item.get("askPrice", 0)) or 0)
        if last_price <= 0:
            return
        self._last_snapshots[symbol.upper().replace("/", "").replace(":", "")] = {
            "pair": normalized_pair,
            "timestamp": datetime.utcnow(),
            "last_price": last_price,
            "volume_24h": float(item.get("q", item.get("quoteVolume", 0)) or 0),
            "mark_price": last_price,
            "index_price": last_price,
            "bid_price": bid_price,
            "ask_price": ask_price,
            "bid_size": float(item.get("B", 0) or 0),
            "ask_size": float(item.get("A", 0) or 0),
            "spread": max(ask_price - bid_price, 0.0),
            "funding_rate": 0.0,
            "open_interest": 0.0,
            "source": "binance_ws",
        }
        await self._touch_market_freshness(normalized_pair)

    async def _handle_candle(self, item: dict, tf: str):
        k = item.get("k", {})
        if not k:
            return
        symbol = k.get("s", "")
        normalized_pair = _normalize_pair(symbol)
        candle = {
            "pair": normalized_pair,
            "timeframe": k.get("i", tf),
            "timestamp": datetime.utcfromtimestamp(int(k["t"]) / 1000).replace(tzinfo=None),
            "open": Decimal(str(k.get("o", "0"))),
            "high": Decimal(str(k.get("h", "0"))),
            "low": Decimal(str(k.get("l", "0"))),
            "close": Decimal(str(k.get("c", "0"))),
            "volume": Decimal(str(k.get("v", "0"))),
        }
        values = [float(candle[field]) for field in ("open", "high", "low", "close", "volume")]
        invalid_ohlc = (
            not all(math.isfinite(value) for value in values)
            or candle["high"] < max(candle["open"], candle["close"], candle["low"])
            or candle["low"] > min(candle["open"], candle["close"], candle["high"])
            or candle["volume"] < 0
        )
        if invalid_ohlc:
            logger.warning("Rejected anomalous candle", pair=normalized_pair, timeframe=tf)
            await self._publish_anomaly(normalized_pair, tf, candle["timestamp"], ["invalid_ohlc_or_volume"])
            return
        history = self._return_history.setdefault(key := f"{normalized_pair}:{candle['timeframe']}", [])
        previous_close = self._last_candle_close.get(key)
        if previous_close and previous_close > 0:
            current_close = float(candle["close"])
            current_return = current_close / previous_close - 1.0
            sigma = pstdev(history) if len(history) > 1 else 0.0
            z = abs(current_return - mean(history)) / sigma if sigma > 1e-9 else 0.0
            if abs(current_return) >= 0.20 or (len(history) >= 10 and z > 4.0):
                logger.warning("Rejected price anomaly", pair=normalized_pair, timeframe=tf,
                               return_pct=current_return, zscore=z)
                await self._publish_anomaly(normalized_pair, tf, candle["timestamp"],
                                            ["price_gap_extreme" if abs(current_return) >= .20 else "return_gt_4_sigma"],
                                            {"return": current_return, "zscore": z})
                return
            history.append(current_return)
            if len(history) > 120:
                del history[:-120]
        self._last_candle_close[key] = float(candle["close"])
        if key not in self._candle_buffers:
            self._candle_buffers[key] = []
        self._candle_buffers[key].append(candle)
        await self._touch_market_freshness(normalized_pair)

    async def _publish_anomaly(self, pair: str, timeframe: str, timestamp: datetime,
                               reasons: list[str], details: dict | None = None) -> None:
        try:
            await message_bus.publish(Channels.MARKET_ANOMALY, {
                "pair": pair, "timeframe": timeframe, "timestamp": timestamp.isoformat(),
                "reasons": reasons, "severity": "high", "source": "market-data-gateway",
                "details": details or {},
            })
        except Exception as exc:
            logger.debug("Could not publish candle anomaly: %s", exc)

    async def _persist_candles_loop(self):
        while self._running:
            await asyncio.sleep(5)
            await self._flush_candles()

    async def _flush_candles(self):
        if not self._candle_buffers:
            return
        async with AsyncSessionLocal() as db:
            from sqlalchemy.dialects.postgresql import insert as pg_insert
            for key, candles in list(self._candle_buffers.items()):
                if not candles:
                    continue
                seen = set()
                unique_candles = []
                for c in candles:
                    dedup_key = (c["pair"], c["timeframe"], c["timestamp"])
                    if dedup_key not in seen:
                        seen.add(dedup_key)
                        unique_candles.append(c)
                if not unique_candles:
                    self._candle_buffers[key] = []
                    continue
                try:
                    rows = [{
                        "pair": c["pair"], "timeframe": c["timeframe"],
                        "timestamp": c["timestamp"].replace(tzinfo=None) if c["timestamp"].tzinfo else c["timestamp"],
                        "open": c["open"], "high": c["high"], "low": c["low"],
                        "close": c["close"], "volume": c["volume"], "source": "binance",
                    } for c in unique_candles]
                    stmt = pg_insert(MarketCandle).values(rows).on_conflict_do_nothing(
                        constraint="uq_market_candles_pair_tf_ts"
                    )
                    await db.execute(stmt)
                    await db.commit()
                    logger.info(f"Persisted {len(unique_candles)} candles for {key}")
                except Exception as e:
                    await db.rollback()
                    logger.warning(f"Failed to persist candles: {e}")
                self._candle_buffers[key] = []

    async def _seed_candles_loop(self):
        """Seed market_candles via REST klines (WS candle disabled).
        Fetch 5m klines per symbol, upsert ON CONFLICT DO NOTHING.
        Weight: ~1/symbol ke limit 1500/min — 538 pair ≈ 538 weight/siklus, aman.
        """
        import aiohttp
        import asyncpg
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        interval = int(os.getenv("KLINE_SEED_INTERVAL", "60"))
        # Build symbol list: ALL_SYMBOLS ("BTC/USDT:USDT") → "BTCUSDT"
        symbols = []
        for s in ALL_SYMBOLS:
            s2 = s.upper().strip()
            if "/" in s2:
                symbols.append(s2.split("/")[0] + "USDT")
            else:
                symbols.append(s2.replace(":", ""))
        symbols = sorted(set(symbols))
        logger.info(f"Kline seeder started (interval={interval}s, {len(symbols)} symbols)")

        while self._running:
            try:
                rows = []
                async with aiohttp.ClientSession() as session:
                    for sym in symbols:
                        url = f"https://fapi.binance.com/fapi/v1/klines?symbol={sym}&interval=5m&limit=2"
                        try:
                            async with session.get(url, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                                if resp.status != 200:
                                    continue
                                data = await resp.json()
                                for k in data:
                                    rows.append({
                                        "pair": _normalize_pair(sym),
                                        "timeframe": "5m",
                                        "timestamp": datetime.utcfromtimestamp(int(k[0]) / 1000).replace(tzinfo=None),
                                        "open": Decimal(str(k[1])),
                                        "high": Decimal(str(k[2])),
                                        "low": Decimal(str(k[3])),
                                        "close": Decimal(str(k[4])),
                                        "volume": Decimal(str(k[5])),
                                        "source": "binance_rest_kline",
                                    })
                        except Exception as e:
                            logger.warning(f"kline fetch fail {sym}: {e}")
                        await asyncio.sleep(0.02)
                if rows:
                    async with AsyncSessionLocal() as db:
                        try:
                            stmt = pg_insert(MarketCandle).values(rows).on_conflict_do_nothing(
                                constraint="uq_market_candles_pair_tf_ts"
                            )
                            await db.execute(stmt)
                            await db.commit()
                            logger.info(f"Kline seeder: {len(rows)} rows upserted ({len(symbols)} symbols)")
                        except Exception as e:
                            await db.rollback()
                            logger.warning(f"Kline seeder persist failed: {e}")
            except Exception as e:
                logger.error(f"Kline seeder cycle error: {e}")
            await asyncio.sleep(interval)

    async def _persist_snapshots_loop(self):
        while self._running:
            await asyncio.sleep(30)
            await self._flush_snapshots()

    async def _flush_snapshots(self):
        if not self._last_snapshots:
            return
        async with AsyncSessionLocal() as db:
            for _symbol, snap in list(self._last_snapshots.items()):
                db_obj = MarketSnapshot(
                    pair=snap["pair"], timestamp=snap["timestamp"],
                    mark_price=snap["mark_price"], index_price=snap["index_price"],
                    last_price=snap["last_price"], bid_price=snap["bid_price"],
                    ask_price=snap["ask_price"], bid_size=snap["bid_size"],
                    ask_size=snap["ask_size"], spread=snap["spread"],
                    funding_rate=snap["funding_rate"], open_interest=snap["open_interest"],
                    volume_24h=snap["volume_24h"], source="binance"
                )
                db.add(db_obj)
            await db.commit()
            self._last_snapshots.clear()

    async def _touch_market_freshness(self, symbol: str):
        """Update Redis freshness key (async-safe sync redis call)."""
        try:
            import redis
            global redis_client
            if redis_client is None:
                redis_client = redis.Redis(
                    host=os.getenv("REDIS_HOST", "redis"),
                    port=int(os.getenv("REDIS_PORT", 6379)),
                    password=os.getenv("REDIS_PASSWORD", "changeme"),
                    decode_responses=True,
                )
            normalized = symbol.upper().split("/")[0].replace(":", "")
            key = f"market:last_update:{normalized}"
            redis_client.set(key, str(datetime.utcnow()), ex=300)
        except Exception as e:
            logger.warning("Failed to update freshness", error=str(e), symbol=symbol)


app = FastAPI()

add_metrics_endpoint(app)
gateway = MarketDataGateway()


@app.on_event("startup")
async def startup_event():
    await gateway.start()


@app.on_event("shutdown")
async def shutdown_event():
    gateway._running = False
    await gateway.obi_monitor.stop()
    await gateway.liq_monitor.stop()
    await close_db()


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "market-data-gateway"}


@app.get("/candles/{pair}/{timeframe}")
async def get_candles(pair: str, timeframe: str, limit: int = 10):
    """Get candles from PostgreSQL."""
    pair = pair.upper()
    candles = await get_candles_async(pair, timeframe, limit)
    return candles


@app.get("/orderbook/{symbol}")
async def get_orderbook_obi(symbol: str):
    snapshot = gateway.obi_monitor.get_snapshot(symbol)
    if not snapshot:
        return {"pair": symbol.upper(), "obi": 0.0, "bid_depth": 0.0, "ask_depth": 0.0, "imbalance": 0.0}
    return snapshot


@app.get("/liquidations")
async def get_liquidations(limit: int = 10):
    return gateway.liq_monitor.get_recent(limit)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
