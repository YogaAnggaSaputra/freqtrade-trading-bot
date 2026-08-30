"""
spread_guard.py
================
Real-Time Dynamic Spread & Volatility Guard — Proteksi Slippage Fatal

Memantau kondisi pasar secara real-time dan memblokir entry order ketika:
  1. Bid-Ask Spread melebar secara abnormal (indikator likuiditas rendah / volatilitas flash)
  2. Volatility Velocity terlalu tinggi (ATR naik mendadak dalam waktu singkat)

Kondisi ini sering terjadi saat:
  - Flash crash / pump mendadak
  - Pergeseran likuiditas sebelum event besar
  - Manipulasi pasar (stop hunt, liquidity grab)

Guards:
  SPREAD_GUARD: spread > SPREAD_MAX_Z_SCORE × std_normal → blokir
  VOLATILITY_GUARD: ATR naik > VOLATILITY_MAX_SURGE_PCT dalam N candle → blokir

Data source: MarketSnapshot dari DB + Candle ATR dari MarketCandle
Fail behavior: FAIL OPEN — jika data tidak tersedia, biarkan trade jalan
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger("risk_gateway.spread_guard")

# ── Configuration ──────────────────────────────────────────────────────────────
SPREAD_GUARD_ENABLED       = os.getenv("SPREAD_GUARD_ENABLED", "true").lower() == "true"
SPREAD_MAX_Z_SCORE         = float(os.getenv("SPREAD_MAX_Z_SCORE", "2.5"))       # Z-score spread > ini = abnormal
SPREAD_LOOKBACK_SNAPSHOTS  = int(os.getenv("SPREAD_LOOKBACK_SNAPSHOTS", "100"))  # Rolling window snapshot
VOLATILITY_GUARD_ENABLED   = os.getenv("VOLATILITY_GUARD_ENABLED", "true").lower() == "true"
VOLATILITY_MAX_SURGE_PCT   = float(os.getenv("VOLATILITY_MAX_SURGE_PCT", "50.0"))  # ATR naik > 50% = blokir
VOLATILITY_CANDLE_LOOKBACK = int(os.getenv("VOLATILITY_CANDLE_LOOKBACK", "3"))    # Cek N candle terakhir


@dataclass
class SpreadGuardResult:
    """Hasil pengecekan spread & volatility guard."""
    passed: bool
    spread_blocked: bool
    volatility_blocked: bool
    spread_z_score: float
    spread_current: float
    spread_mean: float
    atr_current: float
    atr_previous: float
    atr_surge_pct: float
    reason: str
    details: dict[str, Any]


class SpreadGuard:
    """
    Guard real-time untuk deteksi kondisi pasar berbahaya sebelum entry.
    """

    @staticmethod
    async def check(pair: str) -> SpreadGuardResult:
        """
        Jalankan kedua guard (spread + volatility) untuk satu pair.

        Returns SpreadGuardResult dengan passed=False jika ada kondisi berbahaya.
        """
        spread_blocked = False
        vol_blocked = False
        spread_z = 0.0
        spread_cur = 0.0
        spread_mean = 0.0
        atr_cur = 0.0
        atr_prev = 0.0
        atr_surge = 0.0
        reasons = []
        details: dict[str, Any] = {}

        # ── Guard 1: Spread abnormality ────────────────────────────────────────
        if SPREAD_GUARD_ENABLED:
            try:
                spread_z, spread_cur, spread_mean = await SpreadGuard._check_spread(pair)
                details["spread_z_score"] = round(spread_z, 3)
                details["spread_current"] = round(spread_cur, 8)
                details["spread_mean"] = round(spread_mean, 8)

                if spread_z > SPREAD_MAX_Z_SCORE:
                    spread_blocked = True
                    reasons.append(
                        f"Spread ABNORMAL: z={spread_z:.2f} > {SPREAD_MAX_Z_SCORE} "
                        f"(current: {spread_cur:.4f}, mean: {spread_mean:.4f})"
                    )
            except Exception as e:  # noqa: BLE001
                logger.debug("Spread check failed (fail-open): %s", e)

        # ── Guard 2: Volatility surge ──────────────────────────────────────────
        if VOLATILITY_GUARD_ENABLED:
            try:
                atr_cur, atr_prev, atr_surge = await SpreadGuard._check_volatility(pair)
                details["atr_current"] = round(atr_cur, 6)
                details["atr_previous"] = round(atr_prev, 6)
                details["atr_surge_pct"] = round(atr_surge, 2)

                if atr_surge > VOLATILITY_MAX_SURGE_PCT:
                    vol_blocked = True
                    reasons.append(
                        f"Volatility SURGE: ATR naik {atr_surge:.1f}% "
                        f"({atr_prev:.4f} → {atr_cur:.4f}) dalam {VOLATILITY_CANDLE_LOOKBACK} candle"
                    )
            except Exception as e:  # noqa: BLE001
                logger.debug("Volatility check failed (fail-open): %s", e)

        passed = not (spread_blocked or vol_blocked)
        reason = " | ".join(reasons) if reasons else f"Spread & volatility normal for {pair}"

        return SpreadGuardResult(
            passed=passed,
            spread_blocked=spread_blocked,
            volatility_blocked=vol_blocked,
            spread_z_score=spread_z,
            spread_current=spread_cur,
            spread_mean=spread_mean,
            atr_current=atr_cur,
            atr_previous=atr_prev,
            atr_surge_pct=atr_surge,
            reason=reason,
            details=details,
        )

    @staticmethod
    async def _check_spread(pair: str) -> tuple[float, float, float]:
        """
        Hitung z-score spread dari rolling window snapshots.
        Returns: (z_score, current_spread, mean_spread)
        """
        from sqlalchemy import desc, select

        from shared.db.models import MarketSnapshot
        from shared.db.session import AsyncSessionLocal

        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(MarketSnapshot.spread)
                .where(MarketSnapshot.pair == pair)
                .order_by(desc(MarketSnapshot.timestamp))
                .limit(SPREAD_LOOKBACK_SNAPSHOTS)
            )
            spreads_raw = [float(row[0]) for row in result.fetchall() if row[0] is not None]

        if len(spreads_raw) < 10:
            return 0.0, 0.0, 0.0

        spreads = np.array(spreads_raw)
        current_spread = spreads[0]  # Most recent
        mean = float(np.mean(spreads[1:]))  # Exclude current dari mean
        std = float(np.std(spreads[1:]))

        if std == 0:
            return 0.0, current_spread, mean

        z_score = (current_spread - mean) / std
        return float(z_score), current_spread, mean

    @staticmethod
    async def _check_volatility(pair: str) -> tuple[float, float, float]:
        """
        Hitung ATR velocity dari N candle terakhir.
        Returns: (atr_current, atr_previous, surge_pct)
        """
        from sqlalchemy import desc, select

        from shared.db.models import MarketCandle
        from shared.db.session import AsyncSessionLocal

        lookback = VOLATILITY_CANDLE_LOOKBACK + 20  # Extra untuk ATR calculation

        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(MarketCandle)
                .where(MarketCandle.pair == pair, MarketCandle.timeframe == "5m")
                .order_by(desc(MarketCandle.timestamp))
                .limit(lookback)
            )
            candles = result.scalars().all()

        if len(candles) < VOLATILITY_CANDLE_LOOKBACK + 5:
            return 0.0, 0.0, 0.0

        # Hitung ATR sederhana (True Range average)
        def _atr_simple(candle_list, period: int) -> float:
            trs = []
            for i in range(1, len(candle_list)):
                c = candle_list[i]
                prev_c = candle_list[i - 1]
                tr = max(
                    float(c.high) - float(c.low),
                    abs(float(c.high) - float(prev_c.close)),
                    abs(float(c.low) - float(prev_c.close)),
                )
                trs.append(tr)
            return float(np.mean(trs[-period:])) if trs else 0.0

        # ATR sekarang (N candle terakhir) vs ATR sebelumnya
        atr_current = _atr_simple(candles[:VOLATILITY_CANDLE_LOOKBACK + 5], VOLATILITY_CANDLE_LOOKBACK)
        atr_previous = _atr_simple(candles[VOLATILITY_CANDLE_LOOKBACK:], VOLATILITY_CANDLE_LOOKBACK)

        if atr_previous == 0:
            return atr_current, atr_previous, 0.0

        surge_pct = ((atr_current - atr_previous) / atr_previous) * 100
        return atr_current, atr_previous, surge_pct
