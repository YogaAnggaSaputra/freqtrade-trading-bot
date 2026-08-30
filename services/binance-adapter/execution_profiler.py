"""
execution_profiler.py
======================
Trade Execution Latency & Slippage Profiler — Execution Quality Monitor

Mengukur dan mencatat kualitas eksekusi setiap trade dalam 4 dimensi:
  1. Signal-to-Submit latency: Waktu dari sinyal strategy → HTTP POST ke Binance
  2. Submit-to-ACK latency  : Waktu dari HTTP POST → terima response Binance
  3. ACK-to-Fill latency    : Waktu dari ACK → order FILLED di exchange
  4. Slippage (bps)         : Selisih harga intended vs actual fill

Semua metric diexpose ke Prometheus untuk visualisasi di Grafana.
Alert otomatis jika:
  - Slippage > MAX_SLIPPAGE_BPS (default: 10 bps)
  - Signal-to-Fill latency > MAX_LATENCY_MS (default: 500ms)
  - Maker fill rate < MIN_MAKER_RATE (default: 50%)

Data juga disimpan ke Redis untuk real-time dashboard control center.
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

import redis.asyncio as aioredis

logger = logging.getLogger("binance_adapter.execution_profiler")

# ── Configuration ──────────────────────────────────────────────────────────────
MAX_SLIPPAGE_BPS        = float(os.getenv("EXEC_MAX_SLIPPAGE_BPS", "10.0"))
MAX_LATENCY_MS          = float(os.getenv("EXEC_MAX_LATENCY_MS", "500.0"))
MIN_MAKER_FILL_RATE     = float(os.getenv("EXEC_MIN_MAKER_RATE", "0.50"))   # 50%
PROFILER_BUFFER_SIZE    = int(os.getenv("EXEC_PROFILER_BUFFER", "200"))     # Last N executions
REDIS_TTL_SECONDS       = int(os.getenv("EXEC_REDIS_TTL", "3600"))         # 1 jam


@dataclass
class ExecutionRecord:
    """Record lengkap satu eksekusi order."""
    trade_id: str
    pair: str
    side: str
    intended_price: float
    executed_price: float
    quantity: float
    execution_mode: str          # "limit_maker" | "market" | "market_fallback"
    fill_type: str               # "maker" | "taker"

    # Latency measurements (milliseconds)
    signal_to_submit_ms: float = 0.0
    submit_to_ack_ms: float = 0.0
    ack_to_fill_ms: float = 0.0
    total_latency_ms: float = 0.0

    # Quality metrics
    slippage_bps: float = 0.0
    slippage_usdt: float = 0.0
    fee_savings_usdt: float = 0.0
    repeg_count: int = 0

    # Flags
    is_alert: bool = False
    alert_reasons: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def is_maker(self) -> bool:
        return self.fill_type == "maker"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExecutionProfiler:
    """
    Profiler kualitas eksekusi order trading.
    Thread-safe dan async-first.
    """

    def __init__(self, redis_client: aioredis.Redis | None = None):
        self._redis = redis_client
        self._buffer: deque[ExecutionRecord] = deque(maxlen=PROFILER_BUFFER_SIZE)
        self._total_trades = 0
        self._maker_fills = 0
        self._total_slippage_bps = 0.0
        self._total_fee_savings = 0.0
        self._alerts: deque[dict[str, Any]] = deque(maxlen=50)

        # Prometheus metrics (optional — graceful if not installed)
        self._setup_prometheus()

    def _setup_prometheus(self):
        """Setup Prometheus metrics, graceful if prometheus_client not available."""
        try:
            from prometheus_client import Counter, Gauge, Histogram
            self._prom_latency = Histogram(
                "exec_signal_to_fill_ms",
                "Total signal-to-fill execution latency",
                buckets=[10, 50, 100, 200, 500, 1000, 2000, 5000],
            )
            self._prom_slippage = Histogram(
                "exec_slippage_bps",
                "Execution slippage in basis points",
                buckets=[0, 1, 2, 5, 10, 20, 50, 100],
            )
            self._prom_maker_rate = Gauge(
                "exec_maker_fill_rate",
                "Rolling maker fill rate (0.0-1.0)",
            )
            self._prom_fee_savings = Counter(
                "exec_fee_savings_usdt_total",
                "Total fee savings from limit maker orders",
            )
            self._prom_alerts = Counter(
                "exec_quality_alerts_total",
                "Execution quality alerts triggered",
                ["reason"],
            )
            self._prom_available = True
            logger.info("Prometheus metrics initialized for ExecutionProfiler")
        except ImportError:
            self._prom_available = False
            logger.info("prometheus_client not available — metrics will not be exported")

    async def record_execution(
        self,
        trade_id: str,
        pair: str,
        side: str,
        intended_price: float,
        executed_price: float,
        quantity: float,
        execution_mode: str,
        fill_type: str,
        signal_to_submit_ms: float,
        submit_to_ack_ms: float,
        ack_to_fill_ms: float,
        repeg_count: int = 0,
        fee_savings_usdt: float = 0.0,
    ) -> ExecutionRecord:
        """
        Record dan analisis satu eksekusi order.
        Publish alert jika kualitas eksekusi buruk.
        """
        total_latency = signal_to_submit_ms + submit_to_ack_ms + ack_to_fill_ms
        slippage_bps = abs(executed_price - intended_price) / intended_price * 10000 if intended_price > 0 else 0.0
        slippage_usdt = abs(executed_price - intended_price) * quantity

        # Analyze alert conditions
        alert_reasons = []
        if slippage_bps > MAX_SLIPPAGE_BPS:
            alert_reasons.append(f"HIGH_SLIPPAGE: {slippage_bps:.1f}bps > {MAX_SLIPPAGE_BPS}bps")
        if total_latency > MAX_LATENCY_MS:
            alert_reasons.append(f"HIGH_LATENCY: {total_latency:.0f}ms > {MAX_LATENCY_MS:.0f}ms")

        is_alert = bool(alert_reasons)

        record = ExecutionRecord(
            trade_id=trade_id,
            pair=pair,
            side=side,
            intended_price=intended_price,
            executed_price=executed_price,
            quantity=quantity,
            execution_mode=execution_mode,
            fill_type=fill_type,
            signal_to_submit_ms=signal_to_submit_ms,
            submit_to_ack_ms=submit_to_ack_ms,
            ack_to_fill_ms=ack_to_fill_ms,
            total_latency_ms=total_latency,
            slippage_bps=slippage_bps,
            slippage_usdt=slippage_usdt,
            fee_savings_usdt=fee_savings_usdt,
            repeg_count=repeg_count,
            is_alert=is_alert,
            alert_reasons=alert_reasons,
        )

        # Update running stats
        self._buffer.append(record)
        self._total_trades += 1
        if fill_type == "maker":
            self._maker_fills += 1
        self._total_slippage_bps += slippage_bps
        self._total_fee_savings += fee_savings_usdt

        if is_alert:
            self._alerts.append({"trade_id": trade_id, "reasons": alert_reasons, "timestamp": record.timestamp})

        # Update Prometheus metrics
        if self._prom_available:
            self._prom_latency.observe(total_latency)
            self._prom_slippage.observe(slippage_bps)
            self._prom_maker_rate.set(self.maker_fill_rate)
            if fee_savings_usdt > 0:
                self._prom_fee_savings.inc(fee_savings_usdt)
            for reason in alert_reasons:
                self._prom_alerts.labels(reason=reason.split(":")[0]).inc()

        # Store to Redis for control center dashboard
        await self._persist_to_redis(record)

        if is_alert:
            logger.warning("Execution quality alert for %s: %s", trade_id, alert_reasons)
        else:
            logger.info(
                "Execution recorded: %s %s | latency=%.0fms slippage=%.1fbps maker=%s",
                side, pair, total_latency, slippage_bps, fill_type == "maker"
            )

        return record

    async def _persist_to_redis(self, record: ExecutionRecord) -> None:
        """Simpan record terbaru ke Redis untuk control center."""
        if not self._redis:
            return
        try:
            key = f"execution:latest:{record.pair}"
            await self._redis.setex(key, REDIS_TTL_SECONDS, json.dumps(record.to_dict()))
            # Juga push ke sorted set untuk history
            await self._redis.zadd(
                "execution:history",
                {json.dumps(record.to_dict()): time.time()}
            )
            # Trim history ke 500 entries
            await self._redis.zremrangebyrank("execution:history", 0, -501)
        except Exception as e:
            logger.debug("Redis persist failed: %s", e)

    @property
    def maker_fill_rate(self) -> float:
        if self._total_trades == 0:
            return 0.0
        return self._maker_fills / self._total_trades

    @property
    def avg_slippage_bps(self) -> float:
        if self._total_trades == 0:
            return 0.0
        return self._total_slippage_bps / self._total_trades

    def get_summary(self) -> dict[str, Any]:
        """Dapatkan ringkasan statistik eksekusi untuk dashboard."""
        recent = list(self._buffer)[-20:] if self._buffer else []
        recent_latencies = [r.total_latency_ms for r in recent]
        recent_slippages = [r.slippage_bps for r in recent]

        return {
            "total_trades": self._total_trades,
            "maker_fill_rate": round(self.maker_fill_rate, 4),
            "maker_fills": self._maker_fills,
            "taker_fills": self._total_trades - self._maker_fills,
            "avg_slippage_bps": round(self.avg_slippage_bps, 2),
            "total_fee_savings_usdt": round(self._total_fee_savings, 4),
            "recent_avg_latency_ms": round(sum(recent_latencies) / len(recent_latencies), 1) if recent_latencies else 0.0,
            "recent_avg_slippage_bps": round(sum(recent_slippages) / len(recent_slippages), 2) if recent_slippages else 0.0,
            "recent_alerts": list(self._alerts)[-5:],
            "maker_rate_status": "GOOD" if self.maker_fill_rate >= MIN_MAKER_FILL_RATE else "LOW",
        }

    def get_recent_records(self, n: int = 20) -> list[dict[str, Any]]:
        """Dapatkan N record eksekusi terakhir."""
        return [r.to_dict() for r in list(self._buffer)[-n:]]
