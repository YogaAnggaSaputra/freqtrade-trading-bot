#!/usr/bin/env python3
"""
Hyperparameter Backtest: Volume Spike & Entry Threshold Optimization
Train: 2026-07-17 to 2026-08-01
Validation: 2026-08-02 to 2026-08-16
"""
import sys
import subprocess
import json
from datetime import datetime

print("=" * 60)
print("HYPEROPT BACKTEST v1")
print("Train: 2026-07-17 -> 2026-08-01")
print("Validate: 2026-08-02 -> 2026-08-16")
print("=" * 60)

# Command untuk hyperopt freqtrade
# Menggunakan strategy AITradingStrategy dengan parameter: volume_spike_factor, adx_threshold, min_conf_long
# Objective: total_profit + profit_median - trade_count_penalty

cmd = """
docker exec deploy-freqtrade-runtime-1 python3 -m freqtrade hyperopt \\
  --strategy AITradingStrategy \\
  --strategy-path /freqtrade/user_data/strategies \\
  --data-dir /freqtrade/user_data/data/binanceusdm/futures \\
  --timerange 20260717-20260801 \\
  --timeframe 5m \\
  --trading-mode futures \\
  --margin-mode isolated \\
  --enable-protections \\
  --max-trades 200 \\
  --spaces buy \\
  --epochs 15 \\
  --print-all \\
  --no-positional \\
  -e 15 \\
  --hyperopt-loss SharpeHyperoptLossDaily \\
  --db-url sqlite:////freqtrade/user_data/tradesv3.sqlite
"""

result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=3600)
print("STDOUT:", result.stdout[-3000:])
print("STDERR:", result.stderr[-2000:] if result.stderr else "None")
print("Return code:", result.returncode)
