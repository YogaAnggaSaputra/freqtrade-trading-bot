"""
shared/messaging/__init__.py
=============================
Redis pub/sub message bus shared across all services.

Usage:
    bus = MessageBus()
    await bus.connect()
    await bus.publish(Channels.MARKET_DATA, {...})
    await bus.subscribe(Channels.ALERT, handler)
    asyncio.create_task(bus.start_listening())
    ...
    await bus.disconnect()
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, Optional

import redis.asyncio as aioredis

from shared.security import get_secret

logger = logging.getLogger("shared.messaging")


class Channels(str, Enum):
    RISK_DECISION = "risk:decision"
    MARKET_DATA = "market:data"
    ORDER_UPDATE = "order:update"
    POSITION_UPDATE = "position:update"
    RECONCILIATION = "reconciliation"
    KILL_SWITCH = "kill:switch"
    ALERT = "alert"
    ORDERBOOK_OBI = "orderbook:obi"   # Real-time Order Book Imbalance signals
    REGIME_UPDATE = "regime:update"   # Market regime classification updates

    # ── Feedback loop pipeline (Fase 1+) ──────────────────────────────
    TRADE_CLOSED = "trade:closed"
    RETRAIN_TRIGGER = "ml:retrain:trigger"
    MODEL_CANDIDATE_READY = "ml:model:candidate_ready"
    MODEL_DEPLOYED = "ml:model:deployed"
    MODEL_REJECTED = "ml:model:rejected"


Handler = Callable[[Dict[str, Any]], Awaitable[None]]


class MessageBus:
    """Async Redis pub/sub with a single pubsub listener task."""

    def __init__(self, host: Optional[str] = None, port: int = 6379,
                 password: Optional[str] = None, decode_responses: bool = True):
        self._host = host or os.getenv("REDIS_HOST", "redis")
        self._port = int(os.getenv("REDIS_PORT", str(port)))
        self._password = password if password is not None else get_secret("redis_password")
        self._decode_responses = decode_responses
        self._client: Optional[aioredis.Redis] = None
        self._pubsub: Optional[aioredis.client.PubSub] = None
        self._handlers: Dict[str, Handler] = {}
        self._listener_task: Optional[asyncio.Task] = None
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        if self._connected:
            return
        self._client = aioredis.Redis(
            host=self._host,
            port=self._port,
            password=self._password,
            decode_responses=self._decode_responses,
        )
        await self._client.ping()
        self._pubsub = self._client.pubsub()
        self._connected = True
        logger.info("MessageBus connected to %s:%s", self._host, self._port)

    async def disconnect(self) -> None:
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except (asyncio.CancelledError, Exception):
                pass
            self._listener_task = None
        if self._pubsub:
            await self._pubsub.close()
            self._pubsub = None
        if self._client:
            await self._client.aclose()
            self._client = None
        self._connected = False
        logger.info("MessageBus disconnected")

    async def publish(self, channel: Channels, payload: Dict[str, Any]) -> None:
        if self._client is None:
            raise RuntimeError("MessageBus not connected. Call connect() first.")
        await self._client.publish(channel.value, json.dumps(payload, default=str))

    async def subscribe(self, channel: Channels, handler: Handler) -> None:
        if self._pubsub is None:
            raise RuntimeError("MessageBus not connected. Call connect() first.")
        await self._pubsub.subscribe(channel.value)
        self._handlers[channel.value] = handler

    async def start_listening(self) -> None:
        if self._pubsub is None or self._listener_task is not None:
            return
        self._listener_task = asyncio.create_task(self._listen_loop())

    async def _listen_loop(self) -> None:
        assert self._pubsub is not None
        while True:
            try:
                message = await self._pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=1.0
                )
            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001
                logger.warning("MessageBus listen error: %s", exc)
                await asyncio.sleep(1.0)
                continue

            if message is None or message.get("type") != "message":
                continue

            channel = message.get("channel")
            handler = self._handlers.get(channel)
            if handler is None:
                continue
            try:
                payload = json.loads(message.get("data", "{}"))
            except (TypeError, json.JSONDecodeError):
                payload = {"raw": message.get("data")}
            try:
                await handler(payload)
            except Exception as exc:  # noqa: BLE001
                logger.warning("MessageBus handler error on %s: %s", channel, exc)


__all__ = ["Channels", "MessageBus", "Handler"]
