"""
limit_chaser.py
================
Post-Only Limit Maker Order Chasing Engine — Fee Savings 60%

Menghemat biaya komisi trading dengan mengutamakan Limit Maker (Post-Only) orders
daripada Market/Taker orders, ketika kondisi pasar memungkinkan.

Fee Savings:
  Binance Futures Taker Fee : 0.05% per trade
  Binance Futures Maker Fee : 0.02% per trade
  Savings per trade         : ~60% fee reduction

Algoritma re-peg (order chasing):
  1. Submit Limit Maker order pada price yang menguntungkan (bid/ask spread)
  2. Tunggu N detik untuk fill
  3. Jika tidak filled, update harga mendekati mid-price (re-peg)
  4. Ulangi max MAX_REPEG kali
  5. Jika masih tidak filled → fallback ke Market order

Kapan menggunakan Limit Maker:
  - Regime: sideways_low_vol ATAU trending dengan momentum rendah (ADX < 25)
  - Spread: normal (tidak di-block oleh SpreadGuard)
  - OBI: tidak ada extreme imbalance (tidak ada sweep)

Kapan fallback ke Market:
  - Cascade/squeeze aktif (butuh eksekusi cepat)
  - Sudah max re-peg attempts
  - Market order mode diset eksplisit

Referensi:
  Binance POST_ONLY TIF: https://developers.binance.com/docs/derivatives/usds-margined-futures
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger("binance_adapter.limit_chaser")

# ── Configuration ──────────────────────────────────────────────────────────────
LIMIT_CHASER_ENABLED        = os.getenv("LIMIT_CHASER_ENABLED", "true").lower() == "true"
LIMIT_CHASER_MAX_WAIT_SEC   = int(os.getenv("LIMIT_CHASER_MAX_WAIT_SECONDS", "30"))
LIMIT_CHASER_MAX_REPEG      = int(os.getenv("LIMIT_CHASER_MAX_REPEG", "5"))
LIMIT_CHASER_REPEG_INTERVAL = int(os.getenv("LIMIT_CHASER_REPEG_INTERVAL_SEC", "6"))
LIMIT_CHASER_PRICE_OFFSET   = float(os.getenv("LIMIT_CHASER_PRICE_OFFSET_TICKS", "1"))  # ticks dari mid
MAKER_ADX_THRESHOLD         = float(os.getenv("MAKER_ADX_THRESHOLD", "25.0"))
MODEL_INFERENCE_URL         = os.getenv("MODEL_INFERENCE_URL", "http://model-inference:8000")


class OrderExecutionMode(str, Enum):
    MARKET  = "MARKET"   # Taker - immediate fill, higher fee
    LIMIT   = "LIMIT"    # Maker - post-only, lower fee, may not fill
    AUTO    = "AUTO"     # Pilih otomatis berdasarkan kondisi pasar


@dataclass
class ChaserResult:
    """Hasil eksekusi oleh LimitChaser."""
    success: bool
    order_id: str | None = None
    executed_price: float | None = None
    executed_qty: float | None = None
    execution_mode: str = "unknown"  # "limit_maker" | "market_fallback"
    repeg_count: int = 0
    time_to_fill_ms: float | None = None
    fill_type: str = "unknown"      # "maker" | "taker" | "unfilled"
    slippage_bps: float = 0.0       # basis points slippage dari intended price
    fee_savings_estimate_usdt: float = 0.0
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class BinanceClient(Protocol):
    """Protocol defining the expected interface of the Binance client."""
    async def place_limit_maker_order(
        self, symbol: str, side: str, quantity: float, price: float,
        client_order_id: str | None = None,
    ) -> dict[str, Any]: ...
    async def place_market_order(
        self, symbol: str, side: str, quantity: float,
        client_order_id: str | None = None,
    ) -> dict[str, Any]: ...
    async def get_order(self, pair: str, order_id: str) -> dict[str, Any]: ...
    async def cancel_order(self, pair: str, order_id: str) -> None: ...


class LimitChaser:
    """
    Mengelola eksekusi order dengan strategi Post-Only Limit Maker.

    Digunakan oleh binance-adapter setiap kali hendak menempatkan order baru.
    Setiap eksekusi dicatat dan dikirim ke ExecutionProfiler.
    """

    def __init__(self, binance_client: BinanceClient | None = None, execution_profiler=None):
        """
        Args:
            binance_client: Client yang bisa place/cancel/check orders di Binance
            execution_profiler: ExecutionProfiler instance untuk tracking metrics
        """
        self._client: BinanceClient | None = binance_client
        self._profiler = execution_profiler

    async def execute(
        self,
        pair: str,
        side: str,            # "BUY" | "SELL"
        quantity: float,
        intended_price: float,
        tick_size: float,     # Minimum price increment untuk pair ini
        regime: str | None = None,
        adx: float | None = None,
        mode: OrderExecutionMode = OrderExecutionMode.AUTO,
        client_order_id: str | None = None,
        trade_id: str = "",
    ) -> ChaserResult:
        """
        Eksekusi order dengan strategi optimal (limit maker atau market).

        Args:
            pair           : Trading pair (BTCUSDT)
            side           : BUY atau SELL
            quantity       : Jumlah yang ingin dibeli/dijual
            intended_price : Harga target (dari sinyal strategi)
            tick_size      : Minimal price increment bursa
            regime         : Market regime dari GMM classifier
            adx            : ADX value dari feature engine
            mode           : Paksa mode tertentu atau AUTO
        """
        start_ts = time.time()

        if not LIMIT_CHASER_ENABLED or mode == OrderExecutionMode.MARKET:
            result = await self._execute_market(
                pair, side, quantity, intended_price, start_ts,
                client_order_id=client_order_id,
            )
        else:
            # Tentukan apakah kondisi cocok untuk limit maker
            use_limit = self._should_use_limit_maker(regime=regime, adx=adx)
            if use_limit or mode == OrderExecutionMode.LIMIT:
                result = await self._execute_limit_with_chasing(
                    pair, side, quantity, intended_price, tick_size, start_ts,
                    client_order_id=client_order_id,
                )
            else:
                result = await self._execute_market(
                    pair, side, quantity, intended_price, start_ts,
                    client_order_id=client_order_id,
                )

        await self._record_profile(
            trade_id=trade_id, pair=pair, side=side, intended_price=intended_price,
            quantity=quantity, result=result,
        )
        return result

    async def _record_profile(self, *, trade_id: str, pair: str, side: str,
                              intended_price: float, quantity: float,
                              result: ChaserResult) -> None:
        if not self._profiler or not result.success or result.executed_price is None:
            return
        try:
            await self._profiler.record_execution(
                trade_id=trade_id or "unknown",
                pair=pair,
                side=side,
                intended_price=intended_price,
                executed_price=float(result.executed_price),
                quantity=quantity,
                execution_mode=result.execution_mode,
                fill_type=result.fill_type,
                signal_to_submit_ms=0.0,
                submit_to_ack_ms=0.0,
                ack_to_fill_ms=float(result.time_to_fill_ms or 0.0),
                repeg_count=result.repeg_count,
                fee_savings_usdt=float(result.fee_savings_estimate_usdt or 0.0),
            )
        except Exception as exc:
            logger.warning("Execution profiler update failed: %s", exc)

    def _should_use_limit_maker(
        self,
        regime: str | None,
        adx: float | None,
    ) -> bool:
        """
        Tentukan apakah kondisi pasar cocok untuk Post-Only limit order.
        Gunakan limit maker ketika pasar tidak terlalu volatile.
        """
        # Sideways market: sangat cocok untuk limit maker
        if regime in ("sideways_low_vol", "sideways_high_vol"):
            return True

        # Trending tapi momentum rendah (ADX rendah): masih bisa limit maker
        if regime in ("trending_up", "trending_down"):
            if adx is not None and adx < MAKER_ADX_THRESHOLD:
                return True
            return False

        # Breakout: hindari limit maker (butuh fill cepat)
        if regime == "breakout":
            return False

        # Default: gunakan limit maker
        return True

    async def _execute_limit_with_chasing(
        self,
        pair: str,
        side: str,
        quantity: float,
        intended_price: float,
        tick_size: float,
        start_ts: float,
        client_order_id: str | None = None,
    ) -> ChaserResult:
        """Kirim limit maker order dan chase harga jika tidak filled."""
        if self._client is None:
            return ChaserResult(
                success=False,
                execution_mode="limit_maker",
                error="No Binance client available",
            )

        repeg_count = 0
        current_price = self._compute_maker_price(intended_price, side, tick_size)
        order_id = None

        for attempt in range(LIMIT_CHASER_MAX_REPEG + 1):
            try:
                # Batal order sebelumnya jika ada
                if order_id:
                    await self._cancel_order(pair, order_id)
                    repeg_count += 1

                # Submit Post-Only Limit order
                order_resp = await self._client.place_limit_maker_order(
                    symbol=pair,
                    side=side,
                    quantity=quantity,
                    price=current_price,
                    client_order_id=client_order_id,
                )
                order_id = order_resp.get("orderId")
                logger.info(
                    "Limit maker order submitted: %s %s @ %.4f (attempt %d/%d)",
                    side, pair, current_price, attempt + 1, LIMIT_CHASER_MAX_REPEG + 1
                )

                # Tunggu fill
                fill_result = await self._wait_for_fill(pair, order_id, LIMIT_CHASER_REPEG_INTERVAL)
                if fill_result:
                    time_to_fill_ms = (time.time() - start_ts) * 1000
                    slippage_bps = abs(fill_result["price"] - intended_price) / intended_price * 10000
                    fee_savings = quantity * fill_result["price"] * 0.0003  # ~60% savings vs taker

                    return ChaserResult(
                        success=True,
                        order_id=order_id,
                        executed_price=fill_result["price"],
                        executed_qty=fill_result["qty"],
                        execution_mode="limit_maker",
                        repeg_count=repeg_count,
                        time_to_fill_ms=time_to_fill_ms,
                        fill_type="maker",
                        slippage_bps=slippage_bps,
                        fee_savings_estimate_usdt=fee_savings,
                    )

                # Tidak filled dalam waktu → update harga mendekati mid
                current_price = self._repeg_price(current_price, intended_price, side, tick_size, attempt)

            except Exception as e:  # noqa: BLE001
                logger.error("Limit maker attempt %d failed: %s", attempt, e)
                break

        # Fallback ke market order
        logger.warning("Max repeg (%d) reached — falling back to market order", LIMIT_CHASER_MAX_REPEG)
        if order_id:
            with contextlib.suppress(Exception):
                await self._cancel_order(pair, order_id)

        return await self._execute_market(
            pair, side, quantity, intended_price, start_ts,
            repeg_count=repeg_count, client_order_id=client_order_id,
        )

    async def _execute_market(
        self,
        pair: str,
        side: str,
        quantity: float,
        intended_price: float,
        start_ts: float,
        repeg_count: int = 0,
        client_order_id: str | None = None,
    ) -> ChaserResult:
        """Eksekusi market order langsung."""
        if self._client is None:
            return ChaserResult(
                success=False,
                execution_mode="market_fallback",
                error="No Binance client available",
            )
        try:
            order_resp = await self._client.place_market_order(
                symbol=pair,
                side=side,
                quantity=quantity,
                client_order_id=client_order_id,
            )
            time_to_fill_ms = (time.time() - start_ts) * 1000
            fill_price = float(order_resp.get("avgPrice", intended_price) or intended_price)
            slippage_bps = abs(fill_price - intended_price) / intended_price * 10000 if intended_price > 0 else 0.0

            return ChaserResult(
                success=True,
                order_id=str(order_resp.get("orderId")),
                executed_price=fill_price,
                executed_qty=float(order_resp.get("executedQty", quantity) or quantity),
                execution_mode="market_fallback" if repeg_count > 0 else "market",
                repeg_count=repeg_count,
                time_to_fill_ms=time_to_fill_ms,
                fill_type="taker",
                slippage_bps=slippage_bps,
                fee_savings_estimate_usdt=0.0,
            )
        except Exception as e:
            return ChaserResult(success=False, execution_mode="market", error=str(e))

    def _compute_maker_price(self, intended_price: float, side: str, tick_size: float) -> float:
        """Hitung harga untuk Post-Only order (sedikit lebih baik dari mid)."""
        offset = tick_size * LIMIT_CHASER_PRICE_OFFSET
        if side.upper() == "BUY":
            return round(intended_price - offset, 8)  # Sedikit di bawah ask
        else:
            return round(intended_price + offset, 8)  # Sedikit di atas bid

    def _repeg_price(
        self,
        current_price: float,
        intended_price: float,
        side: str,
        tick_size: float,
        attempt: int,
    ) -> float:
        """Update harga mendekati intended_price setelah setiap failed attempt."""
        # Aggressively move toward intended price each re-peg
        move_fraction = (attempt + 1) / LIMIT_CHASER_MAX_REPEG
        if side.upper() == "BUY":
            new_price = current_price + (intended_price - current_price) * move_fraction
        else:
            new_price = current_price - (current_price - intended_price) * move_fraction
        return round(new_price / tick_size) * tick_size

    async def _wait_for_fill(self, pair: str, order_id: str, timeout_sec: int) -> dict[str, Any] | None:
        """Poll order status hingga filled atau timeout."""
        if self._client is None:
            logger.warning("No Binance client configured — cannot check order status")
            return None
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            await asyncio.sleep(0.5)
            try:
                status = await self._client.get_order(pair, order_id)
                if status.get("status") == "FILLED":
                    return {
                        "price": float(status.get("avgPrice", 0)),
                        "qty": float(status.get("executedQty", 0)),
                    }
            except Exception as e:
                logger.warning("Order status check failed: %s", e)
        return None

    async def _cancel_order(self, pair: str, order_id: str) -> None:
        """Cancel order yang sedang open."""
        if self._client is None:
            return
        try:
            await self._client.cancel_order(pair, order_id)
        except Exception as e:
            logger.warning("Cancel order failed: %s", e)
