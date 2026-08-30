"""
runner.py
=========
Experiment Runner — menjalankan backtest, walk-forward, dan stress test
menggunakan Freqtrade sebagai engine backtest.

Runner memanggil Freqtrade CLI via subprocess, mengparse output,
dan mengembalikan metrics terstruktur.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from typing import Any

logger = logging.getLogger("experiment_orchestrator.runner")

FREQTRADE_CONFIG_DIR = os.getenv("FREQTRADE_CONFIG_DIR", "/freqtrade/configs")
FREQTRADE_DATA_DIR = os.getenv("FREQTRADE_DATA_DIR", "/freqtrade/user_data")
FREQTRADE_STRATEGY_DIR = os.getenv("FREQTRADE_STRATEGY_DIR", "/freqtrade/strategies")


class BacktestResult:
    def __init__(self, raw: dict[str, Any]):
        self.raw = raw
        self.total_trades: int = raw.get("total_trades", 0)
        self.win_rate: float = raw.get("wins", 0) / max(raw.get("total_trades", 1), 1)
        self.net_profit: float = raw.get("profit_total", 0.0)
        self.max_drawdown: float = abs(raw.get("max_drawdown", 0.0))
        self.profit_factor: float = raw.get("profit_factor", 1.0)
        self.avg_duration_minutes: float = raw.get("holding_avg_s", 0) / 60.0
        self.sharpe: float = raw.get("sharpe", 0.0)
        self.sortino: float = raw.get("sortino", 0.0)
        self.calmar: float = raw.get("calmar", 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_trades": self.total_trades,
            "win_rate": round(self.win_rate, 4),
            "net_profit": round(self.net_profit, 6),
            "max_drawdown": round(self.max_drawdown, 4),
            "profit_factor": round(self.profit_factor, 4),
            "avg_duration_minutes": round(self.avg_duration_minutes, 1),
            "sharpe": round(self.sharpe, 4),
            "sortino": round(self.sortino, 4),
            "calmar": round(self.calmar, 4),
        }

    def passes_minimum_bars(self, min_trades: int = 50) -> bool:
        return self.total_trades >= min_trades


class ExperimentRunner:
    """Menjalankan backtest dan validasi melalui Freqtrade CLI."""

    def __init__(
        self,
        config_base: str = "config.backtest.json",
        strategy: str = "AITradingStrategy",
    ):
        self.config_base = os.path.join(FREQTRADE_CONFIG_DIR, config_base)
        self.strategy = strategy

    async def run_backtest(
        self,
        experiment_id: str,
        candidate_params: dict[str, Any],
        timerange: str = "20240101-20241231",
        pairs: list[str] | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """
        Jalankan backtest satu periode untuk experiment.

        Args:
            experiment_id: ID eksperimen untuk tracking
            candidate_params: parameter yang dimodifikasi (akan di-inject ke config)
            timerange: format YYYYMMDD-YYYYMMDD
            pairs: list pair untuk backtest

        Returns:
            (success, metrics_dict)
        """
        pairs = pairs or ["BTC/USDT:USDT"]
        config_path = await self._write_candidate_config(
            experiment_id, candidate_params, pairs
        )
        if not config_path:
            return False, {"error": "Failed to write candidate config"}

        cmd = [
            "freqtrade", "backtesting",
            "--config", config_path,
            "--strategy", self.strategy,
            "--timerange", timerange,
            "--export", "trades",
            "--export-filename", f"/tmp/backtest_{experiment_id}.json",
        ]

        logger.info("Running backtest for %s: timerange=%s", experiment_id, timerange)
        success, output, stderr = await self._run_command(cmd)

        if not success:
            logger.error("Backtest failed for %s: %s", experiment_id, stderr)
            return False, {"error": stderr[:500], "output": output[:500]}

        metrics = self._parse_backtest_output(output)
        return True, metrics

    async def run_walkforward(
        self,
        experiment_id: str,
        candidate_params: dict[str, Any],
        windows: int = 6,
        train_months: int = 6,
        test_months: int = 1,
        pairs: list[str] | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """
        Walk-forward validation dengan multiple windows.

        Returns:
            (all_passed, aggregated_metrics)
        """
        pairs = pairs or ["BTC/USDT:USDT"]
        window_results = []

        # Hitung time windows (simplified, mulai dari 2023)
        base_year = 2023
        for i in range(windows):
            f"{base_year + i // 12}{(i % 12 + 1):02d}01"
            train_end_month = (i % 12 + train_months) % 12 or 12
            train_end_year = base_year + (i % 12 + train_months) // 12
            test_start = f"{train_end_year}{train_end_month:02d}01"
            test_end_month = (train_end_month + test_months - 1) % 12 + 1
            test_end_year = train_end_year + (train_end_month + test_months - 1) // 12
            test_end = f"{test_end_year}{test_end_month:02d}01"

            timerange = f"{test_start}-{test_end}"
            window_id = f"{experiment_id}_w{i+1}"

            success, metrics = await self.run_backtest(
                window_id, candidate_params, timerange, pairs
            )
            window_results.append({
                "window": i + 1,
                "timerange": timerange,
                "success": success,
                "metrics": metrics,
            })

        # Agregat results
        successful = [r for r in window_results if r["success"]]
        if not successful:
            return False, {"error": "All windows failed", "windows": window_results}

        win_rates = [r["metrics"].get("win_rate", 0) for r in successful]
        drawdowns = [r["metrics"].get("max_drawdown", 1) for r in successful]
        pf_values = [r["metrics"].get("profit_factor", 0) for r in successful]

        aggregated = {
            "windows_total": windows,
            "windows_successful": len(successful),
            "mean_win_rate": round(sum(win_rates) / len(win_rates), 4),
            "max_drawdown": round(max(drawdowns), 4),
            "mean_profit_factor": round(sum(pf_values) / len(pf_values), 4),
            "consistency_score": round(len(successful) / windows, 2),
            "windows": window_results,
        }

        # Pass bila lebih dari 70% window berhasil
        all_passed = aggregated["consistency_score"] >= 0.70
        return all_passed, aggregated

    async def run_stress_test(
        self,
        experiment_id: str,
        base_metrics: dict[str, Any],
        fee_multipliers: list[float] = None,
        slippage_multipliers: list[float] = None,
    ) -> tuple[bool, dict[str, Any]]:
        """
        Stress test: simulasikan kondisi biaya lebih tinggi dan slippage.
        Ini adalah simulasi matematis dari metrics yang ada, bukan re-backtest.
        """
        fee_multipliers = fee_multipliers or [1.0, 1.5, 2.0, 3.0]
        slippage_multipliers = slippage_multipliers or [1.0, 2.0, 3.0]

        base_profit = base_metrics.get("net_profit", 0)
        base_metrics.get("max_drawdown", 0)
        results = []

        for fee_mult in fee_multipliers:
            for slip_mult in slippage_multipliers:
                # Approximate: extra cost = (fee_mult-1)*0.001 + (slip_mult-1)*0.0005 per trade
                trade_count = base_metrics.get("total_trades", 100)
                extra_cost = (
                    (fee_mult - 1) * 0.001 + (slip_mult - 1) * 0.0005
                ) * trade_count
                adjusted_profit = base_profit - extra_cost
                results.append({
                    "fee_multiplier": fee_mult,
                    "slippage_multiplier": slip_mult,
                    "adjusted_net_profit": round(adjusted_profit, 6),
                    "profitable": adjusted_profit > 0,
                })

        pass_count = sum(1 for r in results if r["profitable"])
        pass_rate = pass_count / len(results) if results else 0

        return pass_rate >= 0.60, {
            "pass_rate": round(pass_rate, 4),
            "pass_count": pass_count,
            "total_scenarios": len(results),
            "scenarios": results,
        }

    async def _write_candidate_config(
        self,
        experiment_id: str,
        candidate_params: dict[str, Any],
        pairs: list[str],
    ) -> str | None:
        """Tulis config JSON sementara dengan parameter kandidat."""
        try:
            base_config = {}
            if os.path.exists(self.config_base):
                with open(self.config_base) as f:
                    base_config = json.load(f)

            # Override dengan candidate params
            base_config.update({
                "pair_whitelist": pairs,
                "dry_run": True,
                "experiment_id": experiment_id,
                **candidate_params,
            })

            tmp_path = f"/tmp/candidate_config_{experiment_id}.json"
            with open(tmp_path, "w") as f:
                json.dump(base_config, f, indent=2)
            return tmp_path
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to write candidate config: %s", e)
            return None

    async def _run_command(
        self, cmd: list[str], timeout: int = 3600
    ) -> tuple[bool, str, str]:
        """Jalankan command async dan return (success, stdout, stderr)."""
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
            output = stdout.decode("utf-8", errors="replace")
            err = stderr.decode("utf-8", errors="replace")
            return proc.returncode == 0, output, err
        except TimeoutError:
            logger.error("Command timed out after %ds: %s", timeout, cmd[0])
            return False, "", "Timeout"
        except FileNotFoundError:
            logger.error("freqtrade executable not found; refusing to fabricate backtest metrics")
            return False, "", "freqtrade executable not found"
        except Exception as e:  # noqa: BLE001
            logger.error("Command failed: %s", e)
            return False, "", str(e)

    def _parse_backtest_output(self, output: str) -> dict[str, Any]:
        """Parse output Freqtrade backtest. Cari JSON section."""
        try:
            # Freqtrade --export menghasilkan JSON di file terpisah
            # Di sini kita parse dari stdout summary
            lines = output.split("\n")
            metrics: dict[str, Any] = {}
            for line in lines:
                if "Total trades" in line:
                    parts = line.split("|")
                    if len(parts) > 2:
                        with contextlib.suppress(ValueError):
                            metrics["total_trades"] = int(parts[2].strip())
                elif "Win/Draw/Loss" in line:
                    parts = line.split("|")
                    if len(parts) > 2:
                        wdl = parts[2].strip().split("/")
                        if len(wdl) == 3:
                            try:
                                wins = int(wdl[0].strip())
                                total = metrics.get("total_trades", 1)
                                metrics["win_rate"] = wins / total
                                metrics["wins"] = wins
                            except ValueError:
                                pass
                elif "Profit factor" in line:
                    parts = line.split("|")
                    if len(parts) > 2:
                        with contextlib.suppress(ValueError):
                            metrics["profit_factor"] = float(parts[2].strip())
                elif "Max Drawdown" in line:
                    parts = line.split("|")
                    if len(parts) > 2:
                        try:
                            dd_str = parts[2].strip().replace("%", "")
                            metrics["max_drawdown"] = abs(float(dd_str)) / 100.0
                        except ValueError:
                            pass

            # Defaults bila parsing gagal
            metrics.setdefault("total_trades", 0)
            metrics.setdefault("win_rate", 0.0)
            metrics.setdefault("profit_factor", 0.0)
            metrics.setdefault("max_drawdown", 1.0)
            metrics.setdefault("net_profit", 0.0)
            return metrics
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to parse backtest output: %s", e)
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 1.0,
                "net_profit": 0.0,
                "parse_error": str(e),
            }

    def _mock_backtest_output(self) -> str:
        """Mock output untuk testing (saat freqtrade tidak tersedia)."""
        return (
            "| Total trades          |    120 |\n"
            "| Win/Draw/Loss         |  58/4/58 |\n"
            "| Profit factor         |   1.32 |\n"
            "| Max Drawdown          |   8.5% |\n"
        )
