"""
liquidation_monitor.py
=======================
Liquidation Cascade Detection Engine — "Alpha dari Kehancuran Orang Lain"

Memantau stream `<symbol>@forceOrder` Binance Futures untuk mendeteksi:
  1. Liquidation Cascade: akumulasi likuidasi masif dalam sliding window pendek
  2. Squeeze Signal  : Long Squeeze (short opportunity) / Short Squeeze (long opportunity)
  3. Exhaustion Point: Ketika cascade mulai mereda (counter-trend opportunity)

Data yang dipublish ke Redis channel `liquidation_events`:
  - liquidation_pressure: -1.0 (long squeeze) → +1.0 (short squeeze)
  - cascade_active: bool
  - total_usd_60s: total USD likuidasi dalam 60 detik terakhir
  - exhaustion: bool (cascade melambat setelah besar)

Referensi API:
  wss://fstream.binance.com/ws/btcusdt@forceOrder
  Format: {"e":"forceOrder","E":timestamp,"o":{"s":"BTCUSDT","S":"SELL","o":"LIMIT","f":"IOC","q":"0.014","p":"9425.5","ap":"9425.5","X":"FILLED","l":"0.014","z":"0.014","T":timestamp}}
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("market_data.liquidation_monitor")

# ── Configuration ──────────────────────────────────────────────────────────────
CASCADE_WINDOW_SECONDS = int(os.getenv("LIQUIDATION_CASCADE_WINDOW_SEC", "60"))
CASCADE_THRESHOLD_USD  = float(os.getenv("LIQUIDATION_CASCADE_THRESHOLD_USD", "1_000_000"))   # $1M
EXHAUSTION_RATIO       = float(os.getenv("LIQUIDATION_EXHAUSTION_RATIO", "0.3"))              # 30% dari peak
PRESSURE_DECAY         = float(os.getenv("LIQUIDATION_PRESSURE_DECAY", "0.95"))               # decay per detik

WS_RECONNECT_DELAY = 5  # detik sebelum reconnect setelah error


@dataclass
class LiquidationEvent:
    """Satu event likuidasi yang diterima dari stream Binance."""
    symbol: str
    side: str          # "BUY" = short position diliquidasi (beli paksa) = short squeeze
    qty: float
    price: float
    notional_usd: float
    timestamp_ms: int

    @property
    def is_short_squeeze(self) -> bool:
        """BUY side = short position terliquidasi = tekanan ke atas harga."""
        return self.side.upper() == "BUY"

    @property
    def is_long_squeeze(self) -> bool:
        """SELL side = long position terliquidasi = tekanan ke bawah harga."""
        return self.side.upper() == "SELL"


@dataclass
class CascadeSnapshot:
    """Snapshot kondisi cascade saat ini untuk satu pair."""
    symbol: str
    cascade_active: bool = False
    liquidation_pressure: float = 0.0    # -1.0 (long squeeze) → +1.0 (short squeeze)
    total_usd_window: float = 0.0        # Total USD likuidasi dalam window
    long_liq_usd: float = 0.0           # Total long liquidations USD
    short_liq_usd: float = 0.0          # Total short liquidations USD
    events_in_window: int = 0
    exhaustion: bool = False             # True = cascade mereda, potensi reversal
    exhaustion_type: str = "none"        # "long_exhaustion" | "short_exhaustion" | "none"
    peak_rate_usd_per_min: float = 0.0  # Peak laju likuidasi per menit
    current_rate_usd_per_min: float = 0.0
    last_updated: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "cascade_active": self.cascade_active,
            "liquidation_pressure": round(self.liquidation_pressure, 4),
            "total_usd_window": round(self.total_usd_window, 2),
            "long_liq_usd": round(self.long_liq_usd, 2),
            "short_liq_usd": round(self.short_liq_usd, 2),
            "events_in_window": self.events_in_window,
            "exhaustion": self.exhaustion,
            "exhaustion_type": self.exhaustion_type,
            "peak_rate_usd_per_min": round(self.peak_rate_usd_per_min, 2),
            "current_rate_usd_per_min": round(self.current_rate_usd_per_min, 2),
            "signal": self._compute_signal(),
            "last_updated": self.last_updated,
        }

    def _compute_signal(self) -> str:
        if self.exhaustion:
            return self.exhaustion_type.upper()
        if self.cascade_active:
            if self.liquidation_pressure > 0.3:
                return "SHORT_SQUEEZE_CASCADE"
            elif self.liquidation_pressure < -0.3:
                return "LONG_SQUEEZE_CASCADE"
            return "CASCADE_MIXED"
        return "NORMAL"


class LiquidationMonitor:
    """
    Memonitor stream likuidasi Binance Futures secara real-time.
    Mendeteksi cascade, squeeze, dan exhaustion events.
    """

    def __init__(
        self,
        pairs: list[str],
        testnet: bool = False,
        message_bus=None,
    ):
        self.pairs = [p.upper() for p in pairs]
        self.testnet = testnet
        self.message_bus = message_bus
        self._running = False

        # Rolling event buffer: deque of (timestamp_ms, LiquidationEvent)
        self._event_buffers: dict[str, deque[tuple[float, LiquidationEvent]]] = {
            p: deque() for p in self.pairs
        }

        # Peak tracking untuk exhaustion detection
        self._peak_rates: dict[str, float] = {p: 0.0 for p in self.pairs}

        # Snapshot cache terkini
        self._snapshots: dict[str, CascadeSnapshot] = {
            p: CascadeSnapshot(symbol=p) for p in self.pairs
        }

        self._ws_tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        """Mulai monitoring semua pair."""
        self._running = True
        logger.info("LiquidationMonitor started for pairs: %s", self.pairs)

        # Satu koneksi WS per pair untuk stream forceOrder
        for pair in self.pairs:
            task = asyncio.create_task(self._connect_and_listen(pair))
            self._ws_tasks.append(task)

        # Background: update snapshots & publish events
        asyncio.create_task(self._snapshot_update_loop())

    async def stop(self) -> None:
        """Hentikan semua koneksi."""
        self._running = False
        for task in self._ws_tasks:
            task.cancel()
        logger.info("LiquidationMonitor stopped.")

    def get_snapshot(self, symbol: str) -> dict[str, Any] | None:
        """Ambil snapshot cascade terkini untuk satu pair."""
        snap = self._snapshots.get(symbol.upper())
        return snap.to_dict() if snap else None

    def get_all_snapshots(self) -> dict[str, dict[str, Any]]:
        """Ambil semua snapshot."""
        return {sym: snap.to_dict() for sym, snap in self._snapshots.items()}

    def get_liquidation_pressure(self, symbol: str) -> float:
        """Shortcut: return liquidation_pressure score (-1.0 → +1.0)."""
        snap = self._snapshots.get(symbol.upper())
        return snap.liquidation_pressure if snap else 0.0

    # ── Internal: WebSocket ────────────────────────────────────────────────────

    async def _connect_and_listen(self, pair: str) -> None:
        """Koneksi WS ke stream forceOrder untuk satu pair."""
        import websockets

        stream_name = f"{pair.lower()}@forceOrder"
        if self.testnet:
            url = f"wss://stream.binancefuture.com/ws/{stream_name}"
        else:
            url = f"wss://fstream.binance.com/ws/{stream_name}"

        while self._running:
            try:
                logger.info("Connecting to liquidation stream: %s", url)
                async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                    async for raw_msg in ws:
                        if not self._running:
                            break
                        try:
                            data = json.loads(raw_msg)
                            await self._handle_force_order(data)
                        except Exception as e:  # noqa: BLE001
                            logger.warning("Error parsing liquidation message: %s", e)
            except Exception as e:  # noqa: BLE001
                if self._running:
                    logger.error("Liquidation stream error (%s): %s — reconnecting in %ds", pair, e, WS_RECONNECT_DELAY)
                    await asyncio.sleep(WS_RECONNECT_DELAY)

    async def _handle_force_order(self, data: dict) -> None:
        """Parse dan buffer satu forceOrder event."""
        if data.get("e") != "forceOrder":
            return

        order = data.get("o", {})
        symbol = order.get("s", "")
        if symbol not in self.pairs:
            return

        try:
            qty = float(order.get("q", 0))
            price = float(order.get("ap", order.get("p", 0)))  # average price atau price
            notional_usd = qty * price
            side = order.get("S", "")
            ts_ms = int(order.get("T", time.time() * 1000))

            event = LiquidationEvent(
                symbol=symbol,
                side=side,
                qty=qty,
                price=price,
                notional_usd=notional_usd,
                timestamp_ms=ts_ms,
            )

            now = time.time()
            self._event_buffers[symbol].append((now, event))
            logger.debug("Liquidation: %s %s $%.0f", symbol, side, notional_usd)

        except (ValueError, TypeError) as e:
            logger.warning("Failed to parse forceOrder data: %s | data: %s", e, order)

    # ── Internal: Snapshot Calculation ────────────────────────────────────────

    async def _snapshot_update_loop(self) -> None:
        """Update snapshot setiap 1 detik."""
        while self._running:
            await asyncio.sleep(1.0)
            for pair in self.pairs:
                try:
                    self._update_snapshot(pair)
                    await self._publish_if_cascade(pair)
                except Exception as e:  # noqa: BLE001
                    logger.error("Snapshot update error (%s): %s", pair, e)

    def _update_snapshot(self, symbol: str) -> None:
        """Kalkulasi ulang snapshot dari event buffer."""
        now = time.time()
        cutoff = now - CASCADE_WINDOW_SECONDS
        buf = self._event_buffers[symbol]

        # Hapus event yang sudah expired
        while buf and buf[0][0] < cutoff:
            buf.popleft()

        long_liq = 0.0
        short_liq = 0.0
        for _, event in buf:
            if event.is_long_squeeze:
                long_liq += event.notional_usd
            else:
                short_liq += event.notional_usd

        total = long_liq + short_liq
        events_count = len(buf)

        # Pressure: positif = short squeeze (harga naik), negatif = long squeeze (harga turun)
        pressure = (short_liq - long_liq) / total if total > 0 else 0.0

        # Apply decay pada pressure jika tidak ada events baru
        old_snap = self._snapshots[symbol]
        if events_count == 0 and old_snap.liquidation_pressure != 0.0:
            pressure = old_snap.liquidation_pressure * PRESSURE_DECAY

        cascade_active = total >= CASCADE_THRESHOLD_USD

        # Current rate per menit
        if len(buf) >= 2:
            time_span = buf[-1][0] - buf[0][0]
            current_rate = (total / max(time_span, 1)) * 60
        else:
            current_rate = 0.0

        # Update peak rate
        if current_rate > self._peak_rates[symbol]:
            self._peak_rates[symbol] = current_rate

        # Exhaustion detection: cascade sebelumnya besar, sekarang mereda
        peak_rate = self._peak_rates[symbol]
        exhaustion = False
        exhaustion_type = "none"
        if peak_rate > 0 and current_rate < peak_rate * EXHAUSTION_RATIO and peak_rate > (CASCADE_THRESHOLD_USD / CASCADE_WINDOW_SECONDS * 60):
            exhaustion = True
            exhaustion_type = "short_exhaustion" if old_snap.liquidation_pressure > 0.2 else (
                "long_exhaustion" if old_snap.liquidation_pressure < -0.2 else "mixed_exhaustion"
            )

        # Reset peak bila lama tidak ada event
        if total == 0 and not buf:
            self._peak_rates[symbol] = 0.0

        self._snapshots[symbol] = CascadeSnapshot(
            symbol=symbol,
            cascade_active=cascade_active,
            liquidation_pressure=pressure,
            total_usd_window=total,
            long_liq_usd=long_liq,
            short_liq_usd=short_liq,
            events_in_window=events_count,
            exhaustion=exhaustion,
            exhaustion_type=exhaustion_type,
            peak_rate_usd_per_min=peak_rate,
            current_rate_usd_per_min=current_rate,
            last_updated=now,
        )

    async def _publish_if_cascade(self, symbol: str) -> None:
        """Publish event ke Redis jika cascade atau exhaustion terdeteksi."""
        if self.message_bus is None:
            return

        snap = self._snapshots[symbol]
        if snap.cascade_active or snap.exhaustion:
            try:
                from shared.messaging import Channels
                await self.message_bus.publish(
                    Channels.MARKET_DATA,
                    {
                        "type": "liquidation_cascade",
                        "data": snap.to_dict(),
                    },
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to publish liquidation event: %s", e)
