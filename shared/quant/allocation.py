"""Portfolio exposure controls."""
from __future__ import annotations
from .execution import volatility_parity_weights

def exposure_multiplier(drawdown: float, regime: str) -> float:
    dd = max(0.0, float(drawdown))
    regime_key = str(regime).lower()
    regime_key = {"trending_bull": "trending_up", "trending_bear": "trending_down",
                  "ranging": "sideways_low_vol"}.get(regime_key, regime_key)
    regime_factor = {"trending_up": 1.0, "trending_down": 1.0,
                     "breakout": .8, "sideways_low_vol": .5,
                     "sideways_high_vol": .25, "choppy": .25}.get(regime_key, .5)
    return max(0.0, min(1.0, (1.0 - min(dd / .10, 1.0)) * regime_factor))

def allocate(capital: float, drawdown: float, regime: str, volatilities: dict[str, float]) -> dict:
    weights = volatility_parity_weights(volatilities)
    multiplier = exposure_multiplier(drawdown, regime)
    return {"exposure_multiplier": multiplier,
            "allocations": {p: capital * multiplier * w for p, w in weights.items()}}
