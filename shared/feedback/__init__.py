"""
shared/feedback/__init__.py
===========================
Feedback loop Fase 1: dual-write trade outcomes (DB + Redis publish).

Tujuan:
- Setiap trade close catat ke Postgres sebagai source of truth (durable)
- Publish TRADE_CLOSED ke Redis pub/sub untuk loss-analyzer consume
- Fire-and-forget: gagal TIDAK boleh ganggu main trading loop

Konsumer (loss-analyzer) wajib punya reconciliation sweep karena Redis
pub/sub bersifat at-most-once — pakai `processed_by_attribution=false` di
DB sebagai penanda event yang terlewat.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

import asyncpg
import redis as redis_sync

from shared.security import get_secret

logger = logging.getLogger("shared.feedback")

_REDIS: Optional[redis_sync.Redis] = None
_PG_POOL: Optional[asyncpg.Pool] = None
# Serialisasi write ke postgres — race kalau 2+ trade close bersamaan
# ("another operation is in progress" di asyncpg pool yang sama).
_PG_LOCK: Optional[asyncio.Lock] = None

CHANNEL_TRADE_CLOSED = "trade:closed"


def snapshot_entry_conditions(trade, regime: str, predicted_rr: float,
                              ml_signal: str, ml_prob: float, conf: float,
                              atr_ratio: float, side: str,
                              entry_rate: float, sl_pct: float | None = None,
                              features: dict | None = None) -> None:
    """Simpan kondisi entry ke state trade (in-memory), dibaca saat exit.

    Persist `sl_pct` via CustomDataWrapper (DB) supaya R-multiple konsisten
    lintas restart. Sisa kondisi tetap di fb_entry (in-memory) karena hanya
    dipakai saat trade open. Fail-safe: kalau gagal set attr, diabaikan.
    """
    try:
        if not hasattr(trade, "fb_entry"):
            trade.fb_entry = {}
        trade.fb_entry.update({
            "regime": regime,
            "predicted_rr": predicted_rr,
            "ml_signal": ml_signal,
            "ml_prob": ml_prob,
            "conf_score": conf,
            "atr_ratio": atr_ratio,
            "side": side,
            "entry_rate": entry_rate,
            "features": features or {},
        })
        if sl_pct:
            trade.fb_entry["sl_pct"] = sl_pct
            # NOTE: CDW persist TIDAK di sini — trade.id bisa None saat confirm
            # (belum persist) → data nyasar ke ft_trade_id=0.
            # Persist dilakukan di custom_stoploss(after_fill=True) di strategi.
    except Exception as exc:  # noqa: BLE001
        logger.debug("snapshot_entry_conditions failed: %s", exc)


def _redis() -> redis_sync.Redis:
    """Lazy-init sync redis client (freqtrade main loop bukan asyncio)."""
    global _REDIS
    if _REDIS is None:
        _REDIS = redis_sync.Redis(
            host=os.getenv("REDIS_HOST", "redis"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            password=get_secret("redis_password"),
            socket_timeout=2.0,
            socket_connect_timeout=2.0,
        )
    return _REDIS


async def _pg_pool() -> asyncpg.Pool:
    """Lazy-init asyncpg pool untuk write trade_outcomes."""
    global _PG_POOL
    if _PG_POOL is None:
        pw = get_secret("db_password") or os.getenv("DB_PASSWORD", "changeme")
        _PG_POOL = await asyncpg.create_pool(
            host=os.getenv("DB_HOST", "postgres"),
            port=int(os.getenv("DB_PORT", "5432")),
            user=os.getenv("DB_USER", "botbinance"),
            password=pw,
            database=os.getenv("DB_NAME", "botbinance"),
            min_size=1,
            max_size=3,
            timeout=5.0,
        )
    return _PG_POOL


async def emit_trade_closed(outcome: Dict[str, Any]) -> bool:
    """Dual-write trade outcome: Postgres (durable) + Redis publish.

    Return True jika sukses dua-duanya, False jika salah satu gagal.
    Strategi TIDAK boleh raise dari sini — fire-and-forget.
    """
    db_ok = await _write_trade_outcome(outcome)
    redis_ok = _publish_trade_closed(outcome)
    if not db_ok or not redis_ok:
        logger.warning(
            "emit_trade_closed partial: trade_id=%s db_ok=%s redis_ok=%s",
            outcome.get("trade_id"), db_ok, redis_ok
        )
    return db_ok and redis_ok


async def _write_trade_outcome(outcome: Dict[str, Any]) -> bool:
    """INSERT ON CONFLICT DO NOTHING — idempotent kalau freqtrade re-trigger."""
    global _PG_LOCK
    if _PG_LOCK is None:
        _PG_LOCK = asyncio.Lock()
    try:
        async with _PG_LOCK:  # serialisasi — cegah race pool asyncpg
            pool = await _pg_pool()
            ts_entry = _parse_ts(outcome["timestamp_entry"])
            ts_exit  = _parse_ts(outcome["timestamp_exit"])
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO trade_outcomes
                      (trade_id, pair, timeframe, entry_conditions, exit_reason,
                       pnl_pct, pnl_abs, predicted_rr, actual_rr, regime_at_entry,
                       timestamp_entry, timestamp_exit, processed_by_attribution)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,false)
                    ON CONFLICT (trade_id) DO NOTHING
                    """,
                    int(outcome["trade_id"]),
                    outcome["pair"],
                    outcome["timeframe"],
                    json.dumps(outcome["entry_conditions"], default=str),
                    outcome["exit_reason"],
                    float(outcome["pnl_pct"]),
                    float(outcome["pnl_abs"]),
                    outcome.get("predicted_rr"),
                    outcome.get("actual_rr"),
                    outcome.get("regime_at_entry"),
                    ts_entry,
                    ts_exit,
                )
            return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("write_trade_outcome failed: %s", exc)
        return False


def _parse_ts(value: Any) -> datetime:
    """Accept ISO string atau datetime → datetime."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        s = value.replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    raise ValueError(f"unsupported timestamp type: {type(value)}")


def _publish_trade_closed(outcome: Dict[str, Any]) -> bool:
    """Sync publish ke Redis — tidak blocking kalau Redis mati."""
    try:
        _redis().publish(CHANNEL_TRADE_CLOSED, json.dumps(outcome, default=str))
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("publish TRADE_CLOSED failed: %s", exc)
        return False


__all__ = ["emit_trade_closed", "CHANNEL_TRADE_CLOSED", "snapshot_entry_conditions"]
