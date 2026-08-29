"""Order-book microstructure features with bounded, interpretable outputs."""
from __future__ import annotations

def dom_pressure(bids: float, asks: float) -> float:
    return (float(bids) - float(asks)) / (float(bids) + float(asks) + 1e-12)

def spoofing_score(previous_size: float, current_size: float, traded_size: float) -> float:
    disappeared = max(float(previous_size) - float(current_size), 0.0)
    return min(1.0, disappeared / max(float(previous_size), 1e-9)) if traded_size <= 0 else 0.0

def vpin_like(buy_volume: float, sell_volume: float) -> float:
    return abs(float(buy_volume) - float(sell_volume)) / (float(buy_volume) + float(sell_volume) + 1e-12)

def iceberg_evidence(refill_count: int, displayed_size: float, executed_size: float) -> float:
    if refill_count <= 0 or displayed_size <= 0: return 0.0
    return min(1.0, (float(executed_size) / float(displayed_size)) / max(refill_count, 1) / 5.0)
