"""
adapter.py
===========
Binance USDT-M Futures adapter: auth, REST client, WebSocket streams,
and parsing of Binance payloads into shared schemas.
"""
import asyncio
import json
import os
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import structlog
import websockets
from client import BinanceFuturesClient

from shared.messaging import Channels, MessageBus
from shared.schemas import (
    MarginMode,
    MarketCandle,
    MarketSnapshot,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    PositionSide,
    TimeInForce,
)

logger = structlog.get_logger()

# Binance order status -> shared OrderStatus
ORDER_STATUS_MAP = {
    "NEW": OrderStatus.SUBMITTED,
    "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
    "FILLED": OrderStatus.FILLED,
    "CANCELED": OrderStatus.CLOSED,
    "REJECTED": OrderStatus.RECONCILIATION_REQUIRED,
    "EXPIRED": OrderStatus.CLOSED,
    "EXPIRED_IN_MATCH": OrderStatus.RECONCILIATION_REQUIRED,
}


class BinanceFuturesWSClient:
    """Public market streams + private user data stream."""

    def __init__(self, listen_key: str | None = None, testnet: bool = False):
        self.listen_key = listen_key
        self.testnet = testnet
        self.ws: websockets.WebSocketClientProtocol | None = None
        self._running = False
        self._callbacks: dict[str, Any] = {}
        self._reconnect_task: asyncio.Task | None = None

    def _base_url(self) -> str:
        if self.testnet:
            return "wss://stream.binancefuture.com/ws"
        return "wss://fstream.binance.com/ws"

    def on_message(self, stream: str):
        def decorator(func):
            self._callbacks[stream] = func
            return func
        return decorator

    async def connect(self):
        url = self._base_url()
        if self.listen_key:
            url = f"{url}/{self.listen_key}"
        self.ws = await websockets.connect(url, ping_interval=20, ping_timeout=10)
        self._running = True
        asyncio.create_task(self._listen())

    async def subscribe(self, streams: list[str]):
        if self.ws is None:
            raise RuntimeError("WS not connected")
        # Combined stream endpoint: /stream?streams=a/b/c
        await self.ws.send(json.dumps({
            "method": "SUBSCRIBE",
            "params": streams,
            "id": 1,
        }))

    async def _listen(self):
        try:
            async for message in self.ws:
                data = json.loads(message)
                await self._handle_message(data)
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
            await self._reconnect()

    async def _handle_message(self, data: dict):
        # Combined streams wrap payload in {"stream": "...", "data": {...}}
        if "stream" in data and "data" in data:
            stream = data["stream"]
            payload = data["data"]
            for pattern, callback in self._callbacks.items():
                if pattern in stream or pattern == stream:
                    await callback(payload)
            return
        # Private user data stream: {"e": "ORDER_TRADE_UPDATE", ...}
        if "e" in data:
            event = data["e"]
            for pattern, callback in self._callbacks.items():
                if pattern == event or pattern in event:
                    await callback(data)

    async def _reconnect(self):
        if self._reconnect_task and not self._reconnect_task.done():
            return
        self._reconnect_task = asyncio.create_task(self._do_reconnect())

    async def _do_reconnect(self):
        for attempt in range(10):
            await asyncio.sleep(2 ** attempt)
            try:
                await self.connect()
                return
            except Exception as e:  # noqa: BLE001
                logger.warning("Reconnect attempt %s failed: %s", attempt + 1, e)

    async def close(self):
        self._running = False
        if self.ws:
            await self.ws.close()


class BinanceFuturesAdapter:
    def __init__(self):
        self.rest = BinanceFuturesClient()
        self.ws: BinanceFuturesWSClient | None = None          # public market stream
        self.private_ws: BinanceFuturesWSClient | None = None  # private user data stream
        self.bus = MessageBus()

    async def start(self):
        await self.rest.start()
        # Fix #9: connect message bus BEFORE setting up handlers that call bus.publish()
        await self.bus.connect()

        # Fix #1: public market stream — separate instance
        self.ws = BinanceFuturesWSClient(testnet=self._is_testnet())
        self._setup_ws_handlers()
        await self.ws.connect()

        # Fix #1: private user data stream — separate instance with listenKey
        try:
            listen_key = await self._create_listen_key()
            self.private_ws = BinanceFuturesWSClient(
                listen_key=listen_key, testnet=self._is_testnet()
            )
            self._setup_private_ws_handlers(self.private_ws)
            await self.private_ws.connect()
        except Exception as e:  # noqa: BLE001
            logger.warning("Private user stream not available: %s", e)

    async def stop(self):
        await self.rest.stop()
        if self.ws:
            await self.ws.close()
        if self.private_ws:
            await self.private_ws.close()
        await self.bus.disconnect()

    def _is_testnet(self) -> bool:
        return os.getenv("BINANCE_FUTURES_TESTNET", "false").lower() == "true"

    async def _create_listen_key(self) -> str:
        res = await self.rest._request("POST", "/fapi/v1/listenKey", signed=True)
        return res.get("listenKey", "")

    def _setup_ws_handlers(self):
        """Handlers untuk public market data stream (ticker, kline)."""
        @self.ws.on_message("ticker")
        async def on_ticker(data):
            snapshot = self._parse_ticker(data)
            if snapshot is not None:
                await self.bus.publish(Channels.MARKET_DATA, {"type": "ticker", "data": snapshot.model_dump()})

        @self.ws.on_message("kline")
        async def on_candle(data):
            candle = self._parse_candle(data)
            if candle is not None:
                await self.bus.publish(Channels.MARKET_DATA, {"type": "candle", "data": candle.model_dump()})

    def _setup_private_ws_handlers(self, private_ws: "BinanceFuturesWSClient"):
        """Handlers untuk private user data stream (order updates, account updates)."""
        @private_ws.on_message("ORDER_TRADE_UPDATE")
        async def on_order(data):
            order = self._parse_order_update(data)
            if order is not None:
                await self.bus.publish(Channels.ORDER_UPDATE, {"order": order.model_dump()})

        @private_ws.on_message("ACCOUNT_UPDATE")
        async def on_account(data):
            for pos in self._parse_account_positions(data):
                await self.bus.publish(Channels.POSITION_UPDATE, {"position": pos.model_dump()})

    # ------------------------------------------------------------------
    # Parsers
    # ------------------------------------------------------------------
    def _parse_ticker(self, data: dict) -> MarketSnapshot | None:
        # 24hr ticker: e="24hrTicker", s=symbol, c=lastPrice, b=bid, a=ask,
        # B=bidSize, A=askSize, p=priceChange, P=priceChangePercent, q=quoteVol
        symbol = data.get("s", "")
        if not symbol:
            return None
        ts = data.get("E") or data.get("T")
        last = Decimal(str(data.get("c", "0")))
        bid = Decimal(str(data.get("b", "0")))
        ask = Decimal(str(data.get("a", "0")))
        return MarketSnapshot(
            pair=symbol,
            timestamp=datetime.fromtimestamp(int(ts) / 1000, tz=UTC) if ts else datetime.now(UTC),
            mark_price=Decimal(str(data.get("m", last))),
            index_price=Decimal(str(data.get("i", last))),
            last_price=last,
            bid_price=bid,
            ask_price=ask,
            bid_size=Decimal(str(data.get("B", "0"))),
            ask_size=Decimal(str(data.get("A", "0"))),
            spread=ask - bid,
            funding_rate=Decimal(str(data.get("r", "0"))) if data.get("r") else None,
            open_interest=None,
            volume_24h=Decimal(str(data.get("q", "0"))) if data.get("q") else None,
        )

    def _parse_candle(self, data: dict) -> MarketCandle | None:
        # Kline stream: k = {t,o,h,l,c,v,x}
        k = data.get("k") if isinstance(data, dict) else None
        if not k:
            return None
        return MarketCandle(
            pair=k.get("s", ""),
            timeframe=str(k.get("i", "5m")),
            timestamp=datetime.fromtimestamp(int(k["t"]) / 1000, tz=UTC),
            open=Decimal(str(k.get("o", "0"))),
            high=Decimal(str(k.get("h", "0"))),
            low=Decimal(str(k.get("l", "0"))),
            close=Decimal(str(k.get("c", "0"))),
            volume=Decimal(str(k.get("v", "0"))),
        )

    def _parse_order_update(self, data: dict) -> Order | None:
        o = data.get("o") if isinstance(data, dict) else None
        if not o:
            return None
        status = ORDER_STATUS_MAP.get(o.get("X", "NEW"), OrderStatus.UNKNOWN)
        side = OrderSide.BUY if o.get("S") == "BUY" else OrderSide.SELL
        otype_raw = o.get("o", "LIMIT").upper()
        if otype_raw == "MARKET":
            otype = OrderType.MARKET
        elif otype_raw == "LIMIT":
            otype = OrderType.LIMIT
        elif otype_raw in ("STOP", "STOP_MARKET"):
            otype = OrderType.STOP_MARKET
        elif otype_raw in ("STOP_LIMIT",):
            otype = OrderType.STOP_LIMIT
        elif otype_raw in ("TAKE_PROFIT", "TAKE_PROFIT_MARKET"):
            otype = OrderType.TAKE_PROFIT_MARKET
        else:
            otype = OrderType.LIMIT
        return Order(
            order_id=str(o.get("i", "")),
            client_order_id=o.get("c", str(o.get("i", ""))),
            exchange_order_id=str(o.get("i", "")),
            trade_id=o.get("wt", ""),
            pair=o.get("s", ""),
            side=side,
            order_type=otype,
            status=status,
            amount=Decimal(str(o.get("q", "0"))),
            filled=Decimal(str(o.get("z", "0"))),
            price=Decimal(str(o.get("p", "0"))),
            avg_price=Decimal(str(o.get("ap", "0"))) if o.get("ap") else None,
            stop_price=Decimal(str(o.get("sp", "0"))) if o.get("sp") else None,
            leverage=int(o.get("l", "1")),
            margin_mode=MarginMode.ISOLATED if o.get("mt") == "ISOLATED" else MarginMode.CROSSED,
            time_in_force=TimeInForce(o.get("f", "GTC")) if o.get("f") in ("GTC", "IOC", "FOK") else TimeInForce.GTC,
            created_at=datetime.fromtimestamp(int(o.get("T", 0)) / 1000, tz=UTC) if o.get("T") else datetime.now(UTC),
            updated_at=datetime.now(UTC),
            exchange_timestamp=datetime.fromtimestamp(int(o.get("T", 0)) / 1000, tz=UTC) if o.get("T") else None,
            raw_response=o,
        )

    def _parse_account_positions(self, data: dict) -> list[Position]:
        # ACCOUNT_UPDATE: a = {B:[balances], P:[positions]}
        a = data.get("a") if isinstance(data, dict) else None
        positions = a.get("P", []) if a else []
        out = []
        for p in positions:
            amt = Decimal(str(p.get("pa", "0")))
            if amt == 0:
                continue
            out.append(Position(
                position_id=f"{p.get('s','')}:{p.get('ps','BOTH')}",
                pair=p.get("s", ""),
                side=PositionSide.LONG if amt > 0 else PositionSide.SHORT,
                size=abs(amt),
                entry_price=Decimal(str(p.get("ep", "0"))),
                mark_price=Decimal(str(p.get("mp", "0"))),
                leverage=int(p.get("lev", "1")),
                margin_mode=MarginMode.ISOLATED if p.get("mt") == "ISOLATED" else MarginMode.CROSSED,
                unrealized_pnl=Decimal(str(p.get("up", "0"))),
                realized_pnl=Decimal(str(p.get("rp", "0"))),
                liquidation_price=None,
                margin_ratio=None,
                opened_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
                exchange_position_id=p.get("s", ""),
            ))
        return out

    # ------------------------------------------------------------------
    # Public REST passthrough (used by main.py)
    # ------------------------------------------------------------------
    async def get_ticker(self, symbol: str) -> dict:
        return await self.rest.get_ticker(symbol)

    async def get_candles(self, symbol: str, interval: str, **kwargs) -> list:
        return await self.rest.get_candles(symbol, interval, **kwargs)

    async def get_funding_rate(self, symbol: str) -> dict:
        return await self.rest.get_funding_rate(symbol)

    async def get_account(self) -> dict:
        return await self.rest.get_account()

    async def get_positions(self, symbol: str | None = None) -> list:
        return await self.rest.get_positions(symbol)

    async def get_orders(self, symbol: str) -> list:
        return await self.rest.get_orders(symbol)

    async def get_order_detail(self, order_id: str, symbol: str) -> dict:
        return await self.rest.get_order_detail(order_id, symbol)

    async def place_order(self, **kwargs) -> dict:
        return await self.rest.place_order(**kwargs)

    async def cancel_order(self, order_id: str, symbol: str) -> dict:
        return await self.rest.cancel_order(order_id, symbol)

    async def cancel_all_orders(self, symbol: str) -> dict:
        return await self.rest.cancel_all_orders(symbol)

    async def set_leverage(self, symbol: str, leverage: int) -> dict:
        return await self.rest.set_leverage(symbol, leverage)

    async def set_margin_mode(self, symbol: str, margin_type: str = "ISOLATED") -> dict:
        return await self.rest.set_margin_mode(symbol, margin_type)
