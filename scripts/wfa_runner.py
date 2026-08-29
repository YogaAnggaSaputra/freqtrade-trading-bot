"""Minimal walk-forward split utility for downstream backtest commands."""
from __future__ import annotations
import argparse
from datetime import datetime

def windows(start: str, end: str, train_days: int, test_days: int):
    from datetime import timedelta
    cursor, finish = datetime.fromisoformat(start), datetime.fromisoformat(end)
    while cursor < finish:
        train_end = cursor + timedelta(days=train_days)
        test_end = min(train_end + timedelta(days=test_days), finish)
        if test_end <= train_end: break
        yield cursor, train_end, test_end
        cursor += timedelta(days=test_days)

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--start", required=True); p.add_argument("--end", required=True)
    p.add_argument("--train-days", type=int, default=30); p.add_argument("--test-days", type=int, default=7)
    a = p.parse_args()
    for train_start, train_end, test_end in windows(a.start, a.end, a.train_days, a.test_days):
        print(f"train={train_start.isoformat()}..{train_end.isoformat()} test={train_end.isoformat()}..{test_end.isoformat()}")
