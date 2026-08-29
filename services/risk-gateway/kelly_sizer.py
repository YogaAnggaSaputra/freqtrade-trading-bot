"""
kelly_sizer.py
==============
Fractional Kelly Criterion Position Sizer — Dynamic Capital Allocation

Menggantikan ukuran posisi tetap (fixed 0.5% equity) dengan ukuran dinamis
berdasarkan statistik trading historis menggunakan Fractional Kelly Criterion.

Rumus:
  Full Kelly: f* = (W * R - (1 - W)) / R
    W = Win Rate
    R = Average Win / Average Loss ratio (Risk-Reward Ratio)

  Fractional Kelly: f_actual = f* × fraction (default: 0.25)

Batasan keamanan (immutable via policy):
  - Max size: KELLY_MAX_PCT (default 1.0% equity) — hard cap dari policy
  - Min size: KELLY_MIN_PCT (default 0.1% equity) — tidak terlalu kecil
  - Jika win_rate < 40% atau data < MIN_TRADES → gunakan MIN size (defensive mode)

Referensi:
  - Kelly Criterion: https://en.wikipedia.org/wiki/Kelly_criterion
  - Ed Thorp implementation: "Beat the Dealer" methodology
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from decimal import Decimal

logger = logging.getLogger("risk_gateway.kelly_sizer")

# ── Configuration ──────────────────────────────────────────────────────────────
KELLY_FRACTION      = float(os.getenv("KELLY_FRACTION", "0.25"))       # 25% dari Full Kelly
KELLY_MAX_PCT       = float(os.getenv("KELLY_MAX_PCT", "0.01"))         # Max 1.0% equity
KELLY_MIN_PCT       = float(os.getenv("KELLY_MIN_PCT", "0.001"))        # Min 0.1% equity
KELLY_LOOKBACK      = int(os.getenv("KELLY_LOOKBACK_TRADES", "30"))     # Jumlah trade terakhir
KELLY_MIN_TRADES    = int(os.getenv("KELLY_MIN_TRADES", "10"))          # Min trades untuk aktifkan Kelly
KELLY_ENABLED       = os.getenv("KELLY_ENABLED", "true").lower() == "true"


@dataclass
class KellyResult:
    """Hasil kalkulasi Kelly sizing."""
    size_pct: float           # % dari equity yang disarankan
    size_usdt: float          # Nominal USDT yang disarankan (jika equity diketahui)
    full_kelly_pct: float     # Full Kelly (sebelum fraction)
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    risk_reward_ratio: float
    trade_count: int
    mode: str                 # "kelly" | "fallback_min" | "fallback_insufficient_data" | "disabled"
    reason: str


class KellySizer:
    """
    Menghitung ukuran posisi optimal menggunakan Fractional Kelly Criterion.
    Data statistik diambil dari database TradeDossier.
    """

    @staticmethod
    async def compute(equity: Decimal) -> KellyResult:
        """
        Hitung Kelly size berdasarkan statistik trading terkini.

        Args:
            equity: Total equity saat ini dalam USDT

        Returns:
            KellyResult dengan ukuran posisi yang direkomendasikan
        """
        if not KELLY_ENABLED:
            fallback = float(equity) * KELLY_MIN_PCT
            return KellyResult(
                size_pct=KELLY_MIN_PCT,
                size_usdt=fallback,
                full_kelly_pct=0.0,
                win_rate=0.0,
                avg_win_pct=0.0,
                avg_loss_pct=0.0,
                risk_reward_ratio=0.0,
                trade_count=0,
                mode="disabled",
                reason="Kelly sizing disabled via KELLY_ENABLED=false",
            )

        win_rate, avg_win, avg_loss, trade_count = await KellySizer._fetch_trade_stats()

        # Bayesian Beta-Binomial Updating for conservative win-rate estimation
        from shared.quant.calibration import beta_posterior
        n_wins = int(win_rate * trade_count)
        n_losses = trade_count - n_wins
        post = beta_posterior(n_wins, n_losses)
        conservative_win_rate = post["lower_ci_95"]  # Use lower bound of 95% credible interval

        # Insufficient data → defensive minimum
        if trade_count < KELLY_MIN_TRADES:
            fallback = float(equity) * KELLY_MIN_PCT
            return KellyResult(
                size_pct=KELLY_MIN_PCT,
                size_usdt=fallback,
                full_kelly_pct=0.0,
                win_rate=win_rate,
                avg_win_pct=avg_win,
                avg_loss_pct=avg_loss,
                risk_reward_ratio=0.0,
                trade_count=trade_count,
                mode="fallback_insufficient_data",
                reason=f"Only {trade_count} trades ({KELLY_MIN_TRADES} required) — using min size",
            )

        # Terlalu banyak loss → defensive minimum
        if win_rate < 0.35:
            fallback = float(equity) * KELLY_MIN_PCT
            return KellyResult(
                size_pct=KELLY_MIN_PCT,
                size_usdt=fallback,
                full_kelly_pct=0.0,
                win_rate=win_rate,
                avg_win_pct=avg_win,
                avg_loss_pct=avg_loss,
                risk_reward_ratio=0.0,
                trade_count=trade_count,
                mode="fallback_min",
                reason=f"Win rate {win_rate:.1%} < 35% threshold — defensive minimum size",
            )

        # Hitung Full Kelly dengan conservative Bayesian win rate
        if avg_loss == 0:
            full_kelly = 0.0
        else:
            risk_reward = avg_win / avg_loss
            full_kelly = (conservative_win_rate * risk_reward - (1 - conservative_win_rate)) / risk_reward
            full_kelly = max(0.0, full_kelly)  # Kelly tidak boleh negatif

        # Fractional Kelly
        fractional_kelly = full_kelly * KELLY_FRACTION

        # Clamp ke batas policy
        clamped_pct = max(KELLY_MIN_PCT, min(KELLY_MAX_PCT, fractional_kelly))
        size_usdt = float(equity) * clamped_pct

        reason_parts = [f"Win rate: {win_rate:.1%}", f"RR: {avg_win/avg_loss:.2f}x" if avg_loss > 0 else "RR: N/A",
                        f"Full Kelly: {full_kelly:.2%}", f"Fractional ({KELLY_FRACTION}x): {fractional_kelly:.2%}",
                        f"Clamped to: {clamped_pct:.2%}"]

        return KellyResult(
            size_pct=clamped_pct,
            size_usdt=size_usdt,
            full_kelly_pct=full_kelly,
            win_rate=win_rate,
            avg_win_pct=avg_win,
            avg_loss_pct=avg_loss,
            risk_reward_ratio=avg_win / avg_loss if avg_loss > 0 else 0.0,
            trade_count=trade_count,
            mode="kelly",
            reason=" | ".join(reason_parts),
        )

    @staticmethod
    async def _fetch_trade_stats() -> tuple[float, float, float, int]:
        """
        Ambil statistik win rate dan avg win/loss dari database.
        Returns: (win_rate, avg_win_pct, avg_loss_pct, trade_count)
        """
        try:
            from sqlalchemy import desc, select

            from shared.db.models import TradeDossier
            from shared.db.session import AsyncSessionLocal

            async with AsyncSessionLocal() as db:
                result = await db.execute(
                    select(TradeDossier)
                    .where(TradeDossier.realized_pnl.isnot(None))
                    .order_by(desc(TradeDossier.closed_at))
                    .limit(KELLY_LOOKBACK)
                )
                trades = result.scalars().all()

            if not trades:
                return 0.0, 0.0, 0.0, 0

            wins = [t for t in trades if float(t.realized_pnl) > 0]
            losses = [t for t in trades if float(t.realized_pnl) <= 0]

            win_rate = len(wins) / len(trades)

            # Hitung return pct dari notional
            def pnl_pct(trade) -> float:
                entry = trade.entry or {}
                notional = float(entry.get("notional", 1))
                return abs(float(trade.realized_pnl)) / max(notional, 1)

            avg_win = sum(pnl_pct(t) for t in wins) / len(wins) if wins else 0.0
            avg_loss = sum(pnl_pct(t) for t in losses) / len(losses) if losses else 0.0

            return win_rate, avg_win, avg_loss, len(trades)

        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to fetch trade stats for Kelly: %s", e)
            return 0.0, 0.0, 0.0, 0
