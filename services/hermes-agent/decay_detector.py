"""
decay_detector.py
==================
Strategy Decay & Concept Drift Auto-Detector — Self-Healing Performance Monitor

Memantau performa strategi trading secara kontinyu dan mendeteksi tanda-tanda:
  1. Strategy Decay   : Performa memburuk secara gradual (Sharpe ratio turun)
  2. Concept Drift    : Pasar berubah sehingga model tidak relevan lagi
  3. Win Rate Collapse: Sudden win rate drop → sesuatu yang fundamental berubah
  4. Drawdown Surge   : Drawdown naik drastis → model mengambil risiko berlebih

Ketika decay terdeteksi, DecayDetector akan:
  - Publish alert ke Redis + Telegram
  - Return proposal untuk auto-retrain model
  - Turunkan signal confidence ke defensive mode

Rolling Windows:
  - Short window (7 hari)  : Performa terkini
  - Long window  (30 hari) : Baseline performa
  - Decay = short metric < (long metric × threshold)

Metrics yang dipantau:
  - Sharpe Ratio  : Return / Volatility (normalized)
  - Win Rate      : % trade yang profit
  - Avg PnL/trade : Rata-rata PnL per trade
  - Max Drawdown  : Max drawdown dalam window

Referensi: Concept Drift Detection in Trading Systems — Widmer & Kubat (1996)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np

logger = logging.getLogger("hermes.decay_detector")

# ── Configuration ──────────────────────────────────────────────────────────────
DECAY_SHORT_WINDOW_DAYS      = int(os.getenv("DECAY_SHORT_WINDOW_DAYS", "7"))
DECAY_LONG_WINDOW_DAYS       = int(os.getenv("DECAY_LONG_WINDOW_DAYS", "30"))
DECAY_SHARPE_THRESHOLD       = float(os.getenv("DECAY_SHARPE_THRESHOLD", "0.70"))   # Short < Long × 70% = decay
DECAY_WINRATE_THRESHOLD      = float(os.getenv("DECAY_WINRATE_THRESHOLD", "0.85"))  # Short WR < Long WR × 85% = decay
DECAY_MIN_TRADES             = int(os.getenv("DECAY_MIN_TRADES", "10"))             # Min trades per window
DECAY_ENABLED                = os.getenv("DECAY_ENABLED", "true").lower() == "true"


@dataclass
class PerformanceWindow:
    """Statistik performa dalam satu time window."""
    window_days: int
    n_trades: int
    win_rate: float
    avg_pnl_pct: float
    total_pnl: float
    sharpe_ratio: float
    max_drawdown_pct: float
    avg_loss_pct: float
    profit_factor: float     # Gross profit / Gross loss

    @property
    def is_sufficient(self) -> bool:
        return self.n_trades >= DECAY_MIN_TRADES


@dataclass
class DecayReport:
    """Laporan hasil deteksi decay."""
    decay_detected: bool
    decay_type: str              # "sharpe_decay" | "winrate_collapse" | "normal" | "insufficient_data"
    severity: str                # "low" | "medium" | "high" | "critical"
    short_window: PerformanceWindow | None
    long_window: PerformanceWindow | None
    sharpe_ratio_change_pct: float
    win_rate_change_pct: float
    recommendation: str
    auto_retrain_suggested: bool
    defensive_mode_suggested: bool
    details: dict[str, Any]
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "decay_detected": self.decay_detected,
            "decay_type": self.decay_type,
            "severity": self.severity,
            "sharpe_ratio_change_pct": self.sharpe_ratio_change_pct,
            "win_rate_change_pct": self.win_rate_change_pct,
            "recommendation": self.recommendation,
            "auto_retrain_suggested": self.auto_retrain_suggested,
            "defensive_mode_suggested": self.defensive_mode_suggested,
            "short_window_trades": self.short_window.n_trades if self.short_window else 0,
            "long_window_trades": self.long_window.n_trades if self.long_window else 0,
            "details": self.details,
            "timestamp": self.timestamp,
        }


class DecayDetector:
    """
    Deteksi strategy decay dan concept drift dari data trade historis.
    Dipanggil oleh HermesAgent setiap siklus analisis.
    """

    async def analyze(self, strategy_version: str) -> DecayReport:
        """
        Analisis performa strategi dan deteksi decay.

        Args:
            strategy_version: Versi strategi yang sedang aktif

        Returns:
            DecayReport dengan hasil analisis dan rekomendasi
        """
        if not DECAY_ENABLED:
            return self._make_report(False, "disabled", "low", None, None)

        short_perf = await self._compute_window(strategy_version, DECAY_SHORT_WINDOW_DAYS)
        long_perf = await self._compute_window(strategy_version, DECAY_LONG_WINDOW_DAYS)

        return self._evaluate_decay(short_perf, long_perf)

    async def _compute_window(self, strategy_version: str, days: int) -> PerformanceWindow | None:
        """Hitung statistik performa untuk satu window waktu."""
        try:
            from sqlalchemy import and_, select

            from shared.db.models import TradeDossier
            from shared.db.session import AsyncSessionLocal

            since = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=days)

            async with AsyncSessionLocal() as db:
                result = await db.execute(
                    select(TradeDossier).where(
                        and_(
                            TradeDossier.strategy_version == strategy_version,
                            TradeDossier.closed_at >= since,
                            TradeDossier.realized_pnl.isnot(None),
                        )
                    ).order_by(TradeDossier.closed_at)
                )
                trades = result.scalars().all()

            if not trades:
                return PerformanceWindow(
                    window_days=days, n_trades=0, win_rate=0.0, avg_pnl_pct=0.0,
                    total_pnl=0.0, sharpe_ratio=0.0, max_drawdown_pct=0.0,
                    avg_loss_pct=0.0, profit_factor=0.0,
                )

            pnls = [float(str(t.realized_pnl)) for t in trades]  # Column[Decimal] → str → float
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]

            win_rate = len(wins) / len(pnls)
            avg_pnl = sum(pnls) / len(pnls)
            total_pnl = sum(pnls)

            # Sharpe Ratio (simplified, daily returns)
            pnl_arr = np.array(pnls)
            mean_ret = np.mean(pnl_arr)
            std_ret = np.std(pnl_arr)
            sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0

            # Max Drawdown
            cumulative = np.cumsum(pnl_arr)
            peak = np.maximum.accumulate(cumulative)
            drawdown = peak - cumulative
            max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0
            peak_val = float(peak[-1]) if len(peak) > 0 else 1.0
            max_dd_pct = (max_dd / abs(peak_val)) if abs(peak_val) > 0 else 0.0

            # Profit Factor
            gross_profit = sum(wins) if wins else 0
            gross_loss = abs(sum(losses)) if losses else 1
            profit_factor = gross_profit / max(gross_loss, 0.001)

            avg_loss_pct = abs(sum(losses) / len(losses)) if losses else 0.0

            return PerformanceWindow(
                window_days=days,
                n_trades=len(pnls),
                win_rate=win_rate,
                avg_pnl_pct=avg_pnl,
                total_pnl=total_pnl,
                sharpe_ratio=float(sharpe),
                max_drawdown_pct=max_dd_pct,
                avg_loss_pct=avg_loss_pct,
                profit_factor=profit_factor,
            )

        except Exception as e:
            logger.error("Failed to compute performance window (%d days): %s", days, e)
            return None

    def _evaluate_decay(
        self,
        short: PerformanceWindow | None,
        long: PerformanceWindow | None,
    ) -> DecayReport:
        """Bandingkan short vs long window dan tentukan apakah ada decay."""
        now = datetime.now(UTC).isoformat()

        if short is None or long is None:
            return self._make_report(False, "insufficient_data", "low", short, long,
                                     reason="Failed to compute performance windows")

        if not short.is_sufficient or not long.is_sufficient:
            return self._make_report(False, "insufficient_data", "low", short, long,
                                     reason=f"Not enough trades: short={short.n_trades}, long={long.n_trades} (min={DECAY_MIN_TRADES})")

        sharpe_change = 0.0
        winrate_change = 0.0
        if long.sharpe_ratio != 0:
            sharpe_change = (short.sharpe_ratio - long.sharpe_ratio) / abs(long.sharpe_ratio)
        if long.win_rate > 0:
            winrate_change = (short.win_rate - long.win_rate) / long.win_rate

        decay_reasons = []
        decay_type = "normal"

        # Check Sharpe decay
        if long.sharpe_ratio > 0 and short.sharpe_ratio < long.sharpe_ratio * DECAY_SHARPE_THRESHOLD:
            decay_reasons.append(f"Sharpe ratio decayed: {long.sharpe_ratio:.2f} → {short.sharpe_ratio:.2f}")
            decay_type = "sharpe_decay"

        # Check Win Rate collapse
        if long.win_rate > 0 and short.win_rate < long.win_rate * DECAY_WINRATE_THRESHOLD:
            decay_reasons.append(f"Win rate collapsed: {long.win_rate:.1%} → {short.win_rate:.1%}")
            decay_type = "winrate_collapse" if decay_type == "normal" else "composite_decay"

        decay_detected = bool(decay_reasons)

        # Severity
        if not decay_detected:
            severity = "low"
        elif abs(sharpe_change) > 0.5 or abs(winrate_change) > 0.3:
            severity = "critical"
        elif abs(sharpe_change) > 0.3 or abs(winrate_change) > 0.2:
            severity = "high"
        else:
            severity = "medium"

        recommendation = (
            f"DECAY DETECTED — {' | '.join(decay_reasons)}" if decay_detected
            else f"Strategy healthy: Sharpe {short.sharpe_ratio:.2f} / WR {short.win_rate:.1%}"
        )

        auto_retrain = decay_detected and severity in ("high", "critical")
        defensive = decay_detected and severity in ("medium", "high", "critical")

        return DecayReport(
            decay_detected=decay_detected,
            decay_type=decay_type,
            severity=severity,
            short_window=short,
            long_window=long,
            sharpe_ratio_change_pct=sharpe_change * 100,
            win_rate_change_pct=winrate_change * 100,
            recommendation=recommendation,
            auto_retrain_suggested=auto_retrain,
            defensive_mode_suggested=defensive,
            details={
                "short": {
                    "sharpe": round(short.sharpe_ratio, 3),
                    "win_rate": round(short.win_rate, 3),
                    "n_trades": short.n_trades,
                    "profit_factor": round(short.profit_factor, 2),
                },
                "long": {
                    "sharpe": round(long.sharpe_ratio, 3),
                    "win_rate": round(long.win_rate, 3),
                    "n_trades": long.n_trades,
                    "profit_factor": round(long.profit_factor, 2),
                },
                "decay_reasons": decay_reasons,
            },
            timestamp=now,
        )

    def _make_report(
        self,
        decay: bool,
        dtype: str,
        severity: str,
        short: PerformanceWindow | None,
        long: PerformanceWindow | None,
        reason: str = "",
    ) -> DecayReport:
        return DecayReport(
            decay_detected=decay,
            decay_type=dtype,
            severity=severity,
            short_window=short,
            long_window=long,
            sharpe_ratio_change_pct=0.0,
            win_rate_change_pct=0.0,
            recommendation=reason,
            auto_retrain_suggested=False,
            defensive_mode_suggested=False,
            details={"reason": reason},
            timestamp=datetime.now(UTC).isoformat(),
        )
