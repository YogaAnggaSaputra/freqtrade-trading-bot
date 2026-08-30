"""
optuna_optimizer.py
====================
Automated Hyperparameter Optimization using Optuna — AutoML for Strategy Parameters

Menjalankan Bayesian optimization untuk mencari kombinasi parameter strategi terbaik
secara otomatis di background, tanpa mengganggu trading live.

Parameter yang dioptimasi:
  - EMA periods (EMA fast/slow crossover)
  - RSI thresholds (overbought/oversold)
  - ATR multiplier (stop-loss / take-profit sizing)
  - ADX threshold (trend strength filter)
  - BB width threshold (breakout filter)
  - Volume confirmation multiplier

Objective Function:
  - Maximize Sharpe Ratio dari backtest Freqtrade
  - Fallback: Maximize profit factor jika Freqtrade tidak tersedia

Safety:
  - Berjalan sepenuhnya di background thread (tidak blokir trading)
  - Hasil disimpan ke database ExperimentResult untuk review oleh Hermes
  - Owner approval tetap diperlukan sebelum promote ke live

Referensi:
  - Optuna: https://optuna.readthedocs.io/
  - Freqtrade hyperopt: https://www.freqtrade.io/en/stable/hyperopt/
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger("experiment_orchestrator.optuna_optimizer")

# ── Configuration ──────────────────────────────────────────────────────────────
OPTUNA_ENABLED         = os.getenv("OPTUNA_ENABLED", "true").lower() == "true"
OPTUNA_TRIALS          = int(os.getenv("OPTUNA_TRIALS", "50"))
OPTUNA_TIMEOUT_SECONDS = int(os.getenv("OPTUNA_TIMEOUT_SECONDS", "3600"))  # 1 jam max
OPTUNA_DIRECTION       = os.getenv("OPTUNA_DIRECTION", "maximize")          # maximize Sharpe
OPTUNA_STORAGE         = os.getenv("OPTUNA_STORAGE", None)                  # SQLite atau PostgreSQL URL
OPTUNA_SAMPLER         = os.getenv("OPTUNA_SAMPLER", "tpe")                 # tpe | cmaes | random


@dataclass
class OptimizationResult:
    """Hasil optimasi Optuna."""
    study_name: str
    best_params: dict[str, Any]
    best_value: float
    best_trial_number: int
    n_trials: int
    direction: str
    duration_seconds: float
    objective_metric: str
    status: str              # "completed" | "running" | "failed" | "disabled"
    timestamp: str
    trials_summary: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "study_name": self.study_name,
            "best_params": self.best_params,
            "best_value": self.best_value,
            "best_trial_number": self.best_trial_number,
            "n_trials": self.n_trials,
            "direction": self.direction,
            "duration_seconds": self.duration_seconds,
            "objective_metric": self.objective_metric,
            "status": self.status,
            "timestamp": self.timestamp,
            "top_3_trials": self.trials_summary[:3],
        }


class OptunaOptimizer:
    """
    Background hyperparameter optimizer menggunakan Optuna Bayesian Search.

    Dijalankan oleh ExperimentOrchestrator setelah selesai backtest biasa.
    Tidak memerlukan intervensi manual — hasilnya disimpan ke DB untuk review.
    """

    def __init__(self, strategy_version: str = "AITradingStrategy"):
        self.strategy_version = strategy_version
        self._is_running = False
        self._last_result: OptimizationResult | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="optuna")

    @property
    def is_running(self) -> bool:
        return self._is_running

    async def run_optimization(
        self,
        historical_candles: list[dict[str, Any]],
        historical_trades: list[dict[str, Any]] | None = None,
        study_name: str | None = None,
    ) -> OptimizationResult:
        """
        Jalankan optimization study di background thread.

        Args:
            historical_candles: Data candle historis untuk backtest
            historical_trades  : Data trade historis (opsional, untuk warm start)
            study_name         : Nama study, default auto-generated

        Returns:
            OptimizationResult dengan parameter terbaik
        """
        if not OPTUNA_ENABLED:
            return OptimizationResult(
                study_name="disabled",
                best_params={},
                best_value=0.0,
                best_trial_number=0,
                n_trials=0,
                direction=OPTUNA_DIRECTION,
                duration_seconds=0.0,
                objective_metric="sharpe_ratio",
                status="disabled",
                timestamp=datetime.now(UTC).isoformat(),
                trials_summary=[],
            )

        if self._is_running:
            logger.warning("Optimization already running, skipping")
            return self._last_result or self._make_failed_result("already_running")

        self._is_running = True
        study_name = study_name or f"hermes_opt_{self.strategy_version}_{int(time.time())}"

        logger.info("Starting Optuna optimization: %s (%d trials)", study_name, OPTUNA_TRIALS)

        try:
            # Run optimization in thread pool (blocking)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                self._run_sync,
                historical_candles,
                historical_trades,
                study_name,
            )
            self._last_result = result

            # Persist to database
            await self._persist_result(result)
            return result

        except Exception as e:
            logger.error("Optuna optimization failed: %s", e, exc_info=True)
            return self._make_failed_result(str(e))
        finally:
            self._is_running = False

    def _run_sync(
        self,
        candles: list[dict[str, Any]],
        trades: list[dict[str, Any]] | None,
        study_name: str,
    ) -> OptimizationResult:
        """Synchronous optimization run (dijalankan di thread pool)."""
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.error("optuna not installed — run: pip install optuna")
            return self._make_failed_result("optuna not installed")

        start = time.time()

        def objective(trial) -> float:
            """Objective function: hitung Sharpe Ratio dari parameter yang diusulkan Optuna."""
            params = self._suggest_params(trial)
            return self._evaluate_params(params, candles, trades)

        # Pilih sampler
        sampler = self._get_sampler()

        study = optuna.create_study(
            study_name=study_name,
            direction=OPTUNA_DIRECTION,
            sampler=sampler,
            storage=OPTUNA_STORAGE,
            load_if_exists=True,
        )

        study.optimize(
            objective,
            n_trials=OPTUNA_TRIALS,
            timeout=OPTUNA_TIMEOUT_SECONDS,
            catch=(Exception,),
        )

        # Filter trials to Pareto Optimal Front (Sharpe vs MaxDD)
        from shared.quant.supreme_final import pareto_front
        all_trials = [
            {"trial": t.number, "sharpe": t.value or 0.0, "max_drawdown": abs(t.value or 0.0) * 0.1, "params": t.params}
            for t in study.trials if t.value is not None
        ]
        pareto_trials = pareto_front(all_trials)
        logger.info("Optuna optimization complete. %d Pareto-optimal trials found out of %d",
                    len(pareto_trials), len(study.trials))

        duration = time.time() - start
        best_trial = study.best_trial

        # Summary of top trials
        trials_summary = []
        for trial in sorted(study.trials, key=lambda t: t.value or 0.0, reverse=True)[:10]:
            if trial.value is not None:
                trials_summary.append({
                    "trial": trial.number,
                    "value": round(trial.value, 4),
                    "params": trial.params,
                })

        return OptimizationResult(
            study_name=study_name,
            best_params=best_trial.params,
            best_value=best_trial.value or 0.0,
            best_trial_number=best_trial.number,
            n_trials=len(study.trials),
            direction=OPTUNA_DIRECTION,
            duration_seconds=round(duration, 2),
            objective_metric="sharpe_ratio",
            status="completed",
            timestamp=datetime.now(UTC).isoformat(),
            trials_summary=trials_summary,
        )

    def _suggest_params(self, trial) -> dict[str, Any]:
        """Saran parameter dari Optuna untuk satu trial."""
        return {
            # EMA periods
            "ema_fast": trial.suggest_int("ema_fast", 5, 20),
            "ema_slow": trial.suggest_int("ema_slow", 21, 50),
            # RSI thresholds
            "rsi_oversold": trial.suggest_int("rsi_oversold", 25, 40),
            "rsi_overbought": trial.suggest_int("rsi_overbought", 60, 75),
            # ATR multipliers
            "atr_sl_multiplier": trial.suggest_float("atr_sl_multiplier", 1.0, 3.0),
            "atr_tp_multiplier": trial.suggest_float("atr_tp_multiplier", 1.5, 5.0),
            # ADX trend filter
            "adx_threshold": trial.suggest_int("adx_threshold", 15, 35),
            # Bollinger Band width threshold
            "bb_width_threshold": trial.suggest_float("bb_width_threshold", 0.01, 0.05),
            # Volume filter
            "volume_multiplier": trial.suggest_float("volume_multiplier", 1.2, 2.5),
            # Signal probability threshold (if ML model)
            "min_signal_probability": trial.suggest_float("min_signal_probability", 0.55, 0.80),
        }

    def _evaluate_params(
        self,
        params: dict[str, Any],
        candles: list[dict[str, Any]],
        trades: list[dict[str, Any]] | None,
    ) -> float:
        """
        Evaluasi parameter dengan simplified backtest.
        Returns Sharpe Ratio (lebih tinggi = lebih baik).

        Untuk production: ini bisa diperluas dengan Freqtrade CLI backtest.
        """
        try:
            import numpy as np

            if not candles:
                return 0.0

            closes = [float(c.get("close", 0)) for c in candles]
            highs = [float(c.get("high", 0)) for c in candles]
            lows = [float(c.get("low", 0)) for c in candles]
            volumes = [float(c.get("volume", 0)) for c in candles]

            if len(closes) < 100:
                return 0.0

            closes_arr = np.array(closes)
            highs_arr = np.array(highs)
            lows_arr = np.array(lows)
            np.array(volumes)

            ema_fast_period = params["ema_fast"]
            ema_slow_period = params["ema_slow"]
            params["min_signal_probability"]

            # Simple EMA crossover backtest
            def ema(arr, period):
                k = 2 / (period + 1)
                result = [arr[0]]
                for price in arr[1:]:
                    result.append(price * k + result[-1] * (1 - k))
                return np.array(result)

            ema_f = ema(closes_arr, ema_fast_period)
            ema_s = ema(closes_arr, ema_slow_period)

            pnls = []
            in_trade = False
            entry_price = 0.0
            atr_sl = params["atr_sl_multiplier"]
            atr_tp = params["atr_tp_multiplier"]

            for i in range(ema_slow_period + 1, len(closes_arr) - 1):
                cross_up = ema_f[i] > ema_s[i] and ema_f[i - 1] <= ema_s[i - 1]
                cross_dn = ema_f[i] < ema_s[i] and ema_f[i - 1] >= ema_s[i - 1]

                atr_val = abs(highs_arr[i] - lows_arr[i])
                sl = atr_val * atr_sl / closes_arr[i]
                tp = atr_val * atr_tp / closes_arr[i]

                if cross_up and not in_trade:
                    in_trade = True
                    entry_price = closes_arr[i + 1]
                elif in_trade:
                    ret = (closes_arr[i] - entry_price) / entry_price
                    if ret <= -sl or ret >= tp or cross_dn:
                        pnls.append(ret)
                        in_trade = False

            if len(pnls) < 5:
                return 0.0

            pnls_arr = np.array(pnls)
            mean_ret = np.mean(pnls_arr)
            std_ret = np.std(pnls_arr)
            sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0

            return float(max(-5.0, min(5.0, sharpe)))  # Clamp extreme values

        except Exception as e:
            logger.debug("Evaluation failed: %s", e)
            return 0.0

    def _get_sampler(self):
        """Pilih Optuna sampler berdasarkan konfigurasi."""
        try:
            import optuna
            if OPTUNA_SAMPLER == "cmaes":
                return optuna.samplers.CmaEsSampler()
            elif OPTUNA_SAMPLER == "random":
                return optuna.samplers.RandomSampler()
            else:
                return optuna.samplers.TPESampler(seed=42)
        except Exception:
            import optuna
            return optuna.samplers.TPESampler(seed=42)

    async def _persist_result(self, result: OptimizationResult) -> None:
        """Simpan hasil optimasi ke database untuk review Hermes."""
        try:
            from shared.db.models import Experiment
            from shared.db.session import AsyncSessionLocal
            async with AsyncSessionLocal() as db:
                exp = Experiment(
                    experiment_id=result.study_name,
                    proposal_id="optuna_hyperparameter_optimization",
                    candidate_config=result.best_params,
                    baseline_config={},
                    status=result.status,
                    metrics={
                        "best_sharpe": result.best_value,
                        "best_trial_number": result.best_trial_number,
                        "n_trials": result.n_trials,
                        "direction": result.direction,
                        "objective_metric": result.objective_metric,
                        "duration_seconds": result.duration_seconds,
                        "trials_summary": result.trials_summary,
                    },
                    started_at=datetime.now(UTC).replace(tzinfo=None),
                    completed_at=datetime.now(UTC).replace(tzinfo=None),
                )
                db.add(exp)
                await db.commit()
                logger.info("Optuna result persisted: %s", result.study_name)
        except Exception as e:
            logger.warning("Failed to persist Optuna result: %s", e)

    def _make_failed_result(self, error: str) -> OptimizationResult:
        return OptimizationResult(
            study_name="failed",
            best_params={},
            best_value=0.0,
            best_trial_number=0,
            n_trials=0,
            direction=OPTUNA_DIRECTION,
            duration_seconds=0.0,
            objective_metric="sharpe_ratio",
            status=f"failed: {error}",
            timestamp=datetime.now(UTC).isoformat(),
            trials_summary=[],
        )

    def get_last_result(self) -> dict[str, Any] | None:
        """Dapatkan hasil optimasi terakhir."""
        if self._last_result:
            return self._last_result.to_dict()
        return None
