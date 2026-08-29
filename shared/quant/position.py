"""Deterministic position-health and exit-consensus calculations."""
from __future__ import annotations

def position_health(current_r: float, entry_r: float, regime_now: str,
                    regime_at_entry: str | None, mtf_alignment: float = 1.0) -> dict:
    thesis = not regime_at_entry or regime_now.lower() == regime_at_entry.lower()
    decay = max(0.0, min(1.0, 1.0 - max(entry_r - current_r, 0.0) / max(abs(entry_r) + 1.0, 1.0)))
    score = max(0.0, min(100.0, 100.0 * decay * (0.7 if not thesis else 1.0) * mtf_alignment))
    return {"score": score, "thesis_valid": thesis, "momentum_decay": 1.0 - decay}

def exit_consensus(signals: dict[str, bool], weights: dict[str, float] | None = None,
                   threshold: float = 0.65) -> tuple[bool, float]:
    weights = weights or {"regime": .25, "ml": .20, "momentum": .18,
                          "reversal": .15, "volume": .12, "funding": .10}
    total = sum(float(weights.get(k, 0)) for k, v in signals.items() if v)
    return total >= threshold, total
