"""Weekly hyperopt wrapper with an explicit validation gate.

The script intentionally does not deploy by itself. A candidate must pass the
caller-provided metrics thresholds before an operator promotes it.
"""
from __future__ import annotations
import argparse, json

def passes_gate(metrics: dict, min_calmar: float = 0.0, max_drawdown: float = .20) -> bool:
    return float(metrics.get("calmar", -1e9)) >= min_calmar and abs(float(metrics.get("max_drawdown", 1e9))) <= max_drawdown

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("metrics_json"); parser.add_argument("--min-calmar", type=float, default=0); parser.add_argument("--max-drawdown", type=float, default=.20)
    args = parser.parse_args()
    with open(args.metrics_json, encoding="utf-8") as fh: metrics = json.load(fh)
    print(json.dumps({"approved_for_review": passes_gate(metrics, args.min_calmar, args.max_drawdown), "metrics": metrics}))
