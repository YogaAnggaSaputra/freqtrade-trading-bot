"""
orderbook_monitor.py
======================
Real-time Order Book Imbalance (OBI) Monitor dan Liquidity Hunt Detector.

Membaca WebSocket Binance Futures Depth stream (@depth20@100ms) untuk
setiap pair yang dipantau dan menghitung:

  OBI = (Total Bid Volume - Total Ask Volume) / (Total Bid Volume + Total Ask Volume)
  Range: [-1, 1]
    +1.0 = tekanan beli ekstrem (semua bid, tidak ada ask)
    -1.0 = tekanan jual ekstrem (semua ask, tidak ada bid)
     0.0 = seimbang

Liquidity Sweep Detection:
  - bid_sweep: Volume beli besar menghabiskan semua seller di 3 level teratas
               → ekspektasi harga NAIK
  - ask_sweep: Volume jual besar menghabiskan semua buyer di 3 level teratas
               → ekspektasi harga TURUN

Signal yang dihasilkan:
  STRONG_BUY / BUY / NEUTRAL / SELL / STRONG_SELL

Data OBI di-broadcast ke Redis channel ORDERBOOK_OBI agar bisa dikonsumsi
oleh risk-gateway untuk blokir/izinkan trade berdasarkan tekanan pasar.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger("market_data.orderbook_monitor")

# OBI thresholds
OBI_STRONG_SIGNAL = 0.55   # |OBI| > ini = STRONG_BUY/STRONG_SELL
OBI_SIGNAL = 0.35          # |OBI| > ini = BUY/SELL
OBI_SWEEP_THRESHOLD = 0.70  # Top-3 ratio > ini = liquidity sweep
TOP_LEVELS = 3              # Jumlah level bid/ask untuk sweep detection
DEPTH_LEVELS = 20           # Kedalaman orderbook yang di-subscribe

# History OBI untuk mendeteksi momentum
OBI_HISTORY_SIZE = 10


class OrderBookMonitor:
    """
    WebSocket-based Order Book Imbalance monitor untuk Binance Futures.
    Subscribe ke @depth20@100ms stream untuk setiap pair.
    """

    def __init__(
        self,
        pairs: list[str],
        testnet: bool = False,
        message_bus: Any | None = None,
    ):
        # Normalize pair format: "BTC/USDT:USDT" → "btcusdt"
        self.pairs = [self._normalize_pair(p) for p in pairs]
        self.testnet = testnet
        self.message_bus = message_bus

        # State storage
        self._snapshots: dict[str, dict[str, Any]] = {}
        self._obi_history: dict[str, list[float]] = {p: [] for p in self.pairs}
        self._running = False

        # Build WebSocket URL
        base_ws = (
            "testnet.binancefuture.com" if testnet else "fstream.binance.com"
        )
        streams = [f"{p}@depth{DEPTH_LEVELS}@100ms" for p in self.pairs]
        self._ws_url = f"wss://{base_ws}/stream?streams={'/'.join(streams)}"

    @staticmethod
    def _normalize_pair(pair: str) -> str:
        """Convert pair ke format Binance stream: 'BTC/USDT:USDT' → 'btcusdt'"""
        return pair.lower().replace("/", "").replace(":", "").split("usdt")[0] + "usdt"

    async def start(self) -> None:
        """Mulai monitoring orderbook di background task."""
        self._running = True
        asyncio.create_task(self._run_forever())
        logger.info("OrderBook monitor started for pairs: %s", self.pairs)

    async def stop(self) -> None:
        self._running = False

    async def _run_forever(self) -> None:
        """Loop koneksi dengan auto-reconnect."""
        while self._running:
            try:
                await self._connect_and_listen()
            except Exception as e:
                logger.error("OrderBook WS error: %s — reconnecting in 5s", e)
                if self._running:
                    await asyncio.sleep(5)

    async def _connect_and_listen(self) -> None:
        import websockets  # lazy import
        logger.info("Connecting OrderBook WS: %s", self._ws_url)
        async with websockets.connect(
            self._ws_url,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=5,
        ) as ws:
            async for raw_msg in ws:
                if not self._running:
                    break
                try:
                    data = json.loads(raw_msg)
                    await self._handle_depth_message(data)
                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    logger.warning("OBI message error: %s", e)

    async def _handle_depth_message(self, data: dict[str, Any]) -> None:
        """Parse depth message dan hitung OBI."""
        # Binance combined stream: {"stream": "btcusdt@depth20@100ms", "data": {...}}
        stream = data.get("stream", "")
        payload = data.get("data", data)

        pair_key = stream.split("@")[0].upper() if stream else ""
        if not pair_key:
            return

        bids: list[list[str]] = payload.get("b", [])
        asks: list[list[str]] = payload.get("a", [])

        if not bids and not asks:
            return

        snapshot = self._calculate_obi(pair_key, bids, asks)
        self._snapshots[pair_key] = snapshot

        # Simpan history OBI untuk deteksi momentum
        key_lower = pair_key.lower()
        if key_lower in self._obi_history:
            self._obi_history[key_lower].append(snapshot["obi"])
            if len(self._obi_history[key_lower]) > OBI_HISTORY_SIZE:
                self._obi_history[key_lower].pop(0)

        # Broadcast ke Redis jika ada message_bus
        if self.message_bus and snapshot["signal"] != "NEUTRAL":
            try:
                from shared.messaging import Channels
                await self.message_bus.publish(Channels.ORDERBOOK_OBI, snapshot)
            except Exception as e:
                logger.debug("Failed to publish OBI: %s", e)

    def _calculate_obi(
        self,
        pair: str,
        bids: list[list[str]],
        asks: list[list[str]],
    ) -> dict[str, Any]:
        """
        Hitung Order Book Imbalance dan deteksi liquidity sweep.
        """
        # Parse volume dari level-level orderbook
        bid_volumes = [float(b[1]) for b in bids if len(b) >= 2]
        ask_volumes = [float(a[1]) for a in asks if len(a) >= 2]
        bid_prices = [float(b[0]) for b in bids if len(b) >= 2]
        ask_prices = [float(a[0]) for a in asks if len(a) >= 2]

        total_bid = sum(bid_volumes)
        total_ask = sum(ask_volumes)
        total = total_bid + total_ask

        # OBI utama (seluruh depth)
        obi = (total_bid - total_ask) / total if total > 0 else 0.0

        # Top-3 level OBI untuk liquidity sweep detection
        top_bid = sum(bid_volumes[:TOP_LEVELS])
        top_ask = sum(ask_volumes[:TOP_LEVELS])
        top_total = top_bid + top_ask
        top_ratio = (top_bid - top_ask) / top_total if top_total > 0 else 0.0

        # Spread dalam basis point
        best_bid = bid_prices[0] if bid_prices else 0.0
        best_ask = ask_prices[0] if ask_prices else 0.0
        spread_bps = ((best_ask - best_bid) / best_bid * 10000) if best_bid > 0 else 0.0

        # Deteksi liquidity sweep
        sweep = self._detect_sweep(top_ratio, obi)

        # OBI momentum (apakah OBI sedang naik atau turun?)
        key_lower = pair.lower()
        obi_history = self._obi_history.get(key_lower, [])
        obi_momentum = self._calculate_momentum(obi_history + [obi])

        # Tentukan sinyal trading
        signal = self._classify_signal(obi, sweep, obi_momentum)

        return {
            "pair": pair,
            "obi": round(obi, 4),
            "bid_volume": round(total_bid, 4),
            "ask_volume": round(total_ask, 4),
            "liquidity_sweep": sweep,
            "signal": signal,
            "top3_bid_ratio": round(top_ratio, 4),
            "spread_bps": round(spread_bps, 2),
            "obi_momentum": round(obi_momentum, 4),
            "timestamp": datetime.now(UTC).isoformat(),
        }

    @staticmethod
    def _detect_sweep(top_ratio: float, obi: float) -> str:
        """
        Deteksi liquidity sweep:
        - bid_sweep: bid dominan di level teratas DAN OBI positif kuat
        - ask_sweep: ask dominan di level teratas DAN OBI negatif kuat
        """
        if top_ratio > OBI_SWEEP_THRESHOLD and obi > OBI_SIGNAL:
            return "bid_sweep"   # Big buyer absorbing sellers → expect price UP
        elif top_ratio < -OBI_SWEEP_THRESHOLD and obi < -OBI_SIGNAL:
            return "ask_sweep"  # Big seller absorbing buyers → expect price DOWN
        return "none"

    @staticmethod
    def _calculate_momentum(history: list[float]) -> float:
        """Hitung momentum OBI (perubahan rata-rata dari history ke nilai terkini)."""
        if len(history) < 3:
            return 0.0
        recent_avg = sum(history[-3:]) / 3
        older_avg = sum(history[:-3]) / max(len(history) - 3, 1)
        return recent_avg - older_avg

    @staticmethod
    def _classify_signal(obi: float, sweep: str, momentum: float) -> str:
        """Klasifikasi sinyal trading berdasarkan OBI, sweep, dan momentum."""
        if sweep == "bid_sweep":
            return "STRONG_BUY"
        elif sweep == "ask_sweep":
            return "STRONG_SELL"
        elif obi > OBI_STRONG_SIGNAL or (obi > OBI_SIGNAL and momentum > 0.05):
            return "BUY"
        elif obi < -OBI_STRONG_SIGNAL or (obi < -OBI_SIGNAL and momentum < -0.05):
            return "SELL"
        return "NEUTRAL"

    def get_snapshot(self, pair: str) -> dict[str, Any] | None:
        """Dapatkan snapshot OBI terbaru untuk sebuah pair."""
        clean = pair.upper().replace("/", "").replace(":", "").split("USDT")[0] + "USDT"
        return self._snapshots.get(clean)

    def get_obi(self, pair: str) -> float:
        """Dapatkan nilai OBI terbaru (-1 sampai 1). Default 0.0 jika tidak ada data."""
        snapshot = self.get_snapshot(pair)
        return snapshot["obi"] if snapshot else 0.0

    def get_all_snapshots(self) -> dict[str, dict[str, Any]]:
        return dict(self._snapshots)

    def is_blocking_trade(self, pair: str, side: str) -> bool:
        """
        Cek apakah OBI mengindikasikan kondisi berbahaya untuk trade ini.
        Blokir jika ada liquidity sweep yang berlawanan dengan arah trade.

        Contoh: side=BUY tapi OBI menunjukkan ask_sweep → blokir.
        """
        snapshot = self.get_snapshot(pair)
        if not snapshot:
            return False  # Tidak ada data → tidak blokir

        sweep = snapshot.get("liquidity_sweep", "none")
        obi = snapshot.get("obi", 0.0)
        side_upper = str(side).upper()

        # Sweep berlawanan → jangan masuk
        if side_upper == "BUY" and sweep == "ask_sweep":
            return True
        if side_upper == "SELL" and sweep == "bid_sweep":
            return True

        # OBI ekstrem berlawanan arah → hati-hati
        if side_upper == "BUY" and obi < -0.65:
            return True
        if side_upper == "SELL" and obi > 0.65:
            return True

        return False
