"""
client.py
==========
Binance USDT-M Futures REST client.

Auth: header `X-MBX-APIKEY` + HMAC-SHA256 signature of the query string
(`timestamp` + optional `recvWindow`). No passphrase (unlike Bitget).
Mainnet base URL: https://fapi.binance.com
Testnet base URL: https://testnet.binancefuture.com (via BINANCE_FUTURES_TESTNET=true)
"""
import asyncio
import hashlib
import hmac
import json
import os
import time
from typing import Any

import aiohttp
import structlog
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from shared.security import get_secret

logger = structlog.get_logger()

BASE_URL_MAINNET = "https://fapi.binance.com"
BASE_URL_TESTNET = "https://testnet.binancefuture.com"


class BinanceAPIError(Exception):
    """Raised when Binance returns a non-zero error code."""


class RateLimiter:
    def __init__(self, max_requests: int = 20, window_seconds: float = 1.0):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._timestamps: list[float] = []

    async def acquire(self):
        now = time.time()
        self._timestamps = [ts for ts in self._timestamps if now - ts < self.window_seconds]
        if len(self._timestamps) >= self.max_requests:
            wait_time = self.window_seconds - (now - self._timestamps[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        self._timestamps.append(time.time())


class BinanceFuturesClient:
    def __init__(self):
        self.api_key = get_secret("binance_api_key") or ""
        self.api_secret = get_secret("binance_api_secret") or ""
        testnet = os.getenv("BINANCE_FUTURES_TESTNET", "false").lower() == "true"
        self.base_url = os.getenv(
            "BINANCE_FUTURES_BASE_URL",
            BASE_URL_TESTNET if testnet else BASE_URL_MAINNET,
        )
        self.session: aiohttp.ClientSession = None  # lazy init in start()
        self.rate_limiter = RateLimiter()
        self.ws: Any | None = None
        self.rate_limit_usage: dict[str, int] = {}

    async def start(self):
        if self.session and not self.session.closed:
            return  # already started
        self.session = aiohttp.ClientSession()

    async def stop(self):
        if self.session:
            await self.session.close()
        if self.ws:
            await self.ws.close()

    # ------------------------------------------------------------------
    # Auth / signing
    # ------------------------------------------------------------------
    def _sign(self, query_string: str) -> str:
        return hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _signed_query(self, params: dict[str, Any] | None = None) -> str:
        params = dict(params or {})
        params["timestamp"] = int(time.time() * 1000)
        params.setdefault("recvWindow", 5000)
        qs = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        return f"{qs}&signature={self._sign(qs)}"

    def _headers(self, signed: bool = False) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if signed:
            headers["X-MBX-APIKEY"] = self.api_key
        return headers

    @retry(
        wait=wait_exponential(multiplier=0.5, min=0.5, max=5),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
    )
    async def _request(
        self,
        method: str,
        path: str,
        params: dict | None = None,
        data: dict | None = None,
        signed: bool = False,
    ) -> Any:
        await self.rate_limiter.acquire()
        url = f"{self.base_url}{path}"
        headers = self._headers(signed=signed)

        if signed:
            # Binance signs the QUERY STRING (for both GET and POST).
            qs = self._signed_query(params)
            url = f"{url}?{qs}"
            params = None
            payload = json.dumps(data) if data else None
        else:
            payload = json.dumps(data) if data else None

        if self.session is None or self.session.closed:
            raise RuntimeError("BinanceFuturesClient not started — call await client.start() first")

        async with self.session.request(
            method, url, params=params, data=payload, headers=headers,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            self.rate_limit_usage = {
                key: int(value) for key, value in {
                    "used_weight_1m": resp.headers.get("x-mbx-used-weight-1m", 0),
                    "order_count_10s": resp.headers.get("x-mbx-order-count-10s", 0),
                    "order_count_1m": resp.headers.get("x-mbx-order-count-1m", 0),
                }.items()
            }
            text = await resp.text()
            if resp.status >= 400:
                try:
                    err = json.loads(text)
                    raise BinanceAPIError(
                        f"Binance error {err.get('code')}: {err.get('msg')}"
                    )
                except json.JSONDecodeError:
                    raise BinanceAPIError(f"API error {resp.status}: {text}")
            result = json.loads(text) if text else {}
            # Binance market endpoints may wrap in {"code":..,"msg":..,"data":..}? No —
            # fapi returns plain arrays/objects; error codes come with HTTP 400.
            return result

    def get_rate_limit_status(self) -> dict[str, int]:
        return dict(self.rate_limit_usage)

    # ------------------------------------------------------------------
    # Public market data
    # ------------------------------------------------------------------
    async def get_ticker(self, symbol: str) -> dict:
        return await self._request("GET", "/fapi/v1/ticker/price", params={"symbol": symbol})

    async def get_candles(
        self,
        symbol: str,
        interval: str,
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int = 500,
    ) -> list:
        params: dict[str, Any] = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        return await self._request("GET", "/fapi/v1/klines", params=params)

    async def get_funding_rate(self, symbol: str) -> dict:
        return await self._request("GET", "/fapi/v1/premiumIndex", params={"symbol": symbol})

    async def get_open_interest(self, symbol: str) -> dict:
        return await self._request("GET", "/fapi/v1/openInterest", params={"symbol": symbol})

    async def get_exchange_info(self) -> dict:
        return await self._request("GET", "/fapi/v1/exchangeInfo")

    async def get_depth(self, symbol: str, limit: int = 100) -> dict:
        return await self._request("GET", "/fapi/v1/depth", params={"symbol": symbol, "limit": limit})

    # ------------------------------------------------------------------
    # Account / positions / orders (signed)
    # ------------------------------------------------------------------
    async def get_account(self) -> dict:
        return await self._request("GET", "/fapi/v2/account", signed=True)

    async def get_balance(self) -> list:
        return await self._request("GET", "/fapi/v2/balance", signed=True)

    async def get_positions(self, symbol: str | None = None) -> list:
        params = {}
        if symbol:
            params["symbol"] = symbol
        return await self._request("GET", "/fapi/v2/positionRisk", params=params, signed=True)

    async def get_orders(self, symbol: str) -> list:
        return await self._request("GET", "/fapi/v1/openOrders", params={"symbol": symbol}, signed=True)

    async def get_order_detail(self, order_id: str, symbol: str) -> dict:
        return await self._request(
            "GET", "/fapi/v1/order",
            params={"symbol": symbol, "orderId": order_id}, signed=True,
        )

    async def get_order_history(self, symbol: str, limit: int = 50) -> list:
        return await self._request(
            "GET", "/fapi/v1/allOrders",
            params={"symbol": symbol, "limit": limit}, signed=True,
        )

    async def set_leverage(self, symbol: str, leverage: int) -> dict:
        return await self._request(
            "POST", "/fapi/v1/leverage",
            params={"symbol": symbol, "leverage": leverage}, signed=True,
        )

    async def set_margin_mode(self, symbol: str, margin_type: str = "ISOLATED") -> dict:
        return await self._request(
            "POST", "/fapi/v1/marginType",
            params={"symbol": symbol, "marginType": margin_type}, signed=True,
        )

    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        size: str,
        price: str | None = None,
        leverage: int = 1,
        margin_mode: str = "isolated",
        stop_loss: str | None = None,
        take_profit: str | None = None,
        client_order_id: str | None = None,
        reduce_only: bool = False,
    ) -> dict:
        # Leverage & margin mode are persistent per-symbol settings.
        try:
            await self.set_leverage(symbol, leverage)
            await self.set_margin_mode(symbol, margin_mode.upper())
        except BinanceAPIError as exc:
            logger.warning("set_leverage/margin failed (may already be set)", error=str(exc))

        type_map = {
            "market": "MARKET",
            "limit": "LIMIT",
            "stop_market": "STOP_MARKET",
            "stop_limit": "STOP_LIMIT",
            "take_profit_market": "TAKE_PROFIT_MARKET",
            "take_profit_limit": "TAKE_PROFIT_LIMIT",
        }
        data: dict[str, Any] = {
            "symbol": symbol,
            "side": side.upper(),
            "type": type_map.get(order_type, order_type.upper()),
            "quantity": size,
            "newClientOrderId": client_order_id,
        }
        if price:
            data["price"] = price
        if stop_loss:
            data["stopPrice"] = stop_loss
        if take_profit:
            data["stopPrice"] = take_profit
        if reduce_only:
            data["reduceOnly"] = "true"
        if client_order_id:
            data["newClientOrderId"] = client_order_id

        return await self._request("POST", "/fapi/v1/order", params=data, signed=True)

    async def cancel_order(self, order_id: str, symbol: str) -> dict:
        return await self._request(
            "DELETE", "/fapi/v1/order",
            params={"symbol": symbol, "orderId": order_id}, signed=True,
        )

    async def cancel_all_orders(self, symbol: str) -> dict:
        return await self._request(
            "DELETE", "/fapi/v1/allOpenOrders",
            params={"symbol": symbol}, signed=True,
        )

    # ------------------------------------------------------------------
    # LimitChaser-compatible wrappers
    # These methods match the BinanceClient Protocol in limit_chaser.py
    # ------------------------------------------------------------------

    async def place_limit_maker_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        client_order_id: str | None = None,
    ) -> dict:
        """Post-Only Limit Maker order (timeInForce=GTX). Fee savings ~60%."""
        params: dict[str, Any] = {
            "symbol": symbol,
            "side": side.upper(),
            "type": "LIMIT",
            "timeInForce": "GTX",   # GTX = Post-Only (rejected if would be taker)
            "quantity": str(quantity),
            "price": str(price),
        }
        if client_order_id:
            params["newClientOrderId"] = client_order_id
        return await self._request("POST", "/fapi/v1/order", params=params, signed=True)

    async def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        client_order_id: str | None = None,
    ) -> dict:
        """Market order (taker) — fallback when limit chasing times out."""
        params: dict[str, Any] = {
            "symbol": symbol,
            "side": side.upper(),
            "type": "MARKET",
            "quantity": str(quantity),
        }
        if client_order_id:
            params["newClientOrderId"] = client_order_id
        return await self._request("POST", "/fapi/v1/order", params=params, signed=True)

    async def get_order(self, pair: str, order_id: str) -> dict:
        """Get single order status — LimitChaser polling interface."""
        return await self._request(
            "GET", "/fapi/v1/order",
            params={"symbol": pair, "orderId": order_id}, signed=True,
        )
