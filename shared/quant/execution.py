"""Execution quality and order safety primitives."""
from __future__ import annotations
import math

def slippage_bps(reference: float, fill: float) -> float:
    return abs(float(fill) - float(reference)) / max(abs(float(reference)), 1e-12) * 10000

def choose_order_type(spread_bps: float, slippage_bps_value: float, urgent: bool = False) -> str:
    if urgent or spread_bps > 30 or slippage_bps_value > 25:
        return "market"
    return "limit"

def volatility_parity_weights(volatilities: dict[str, float], max_weight: float = .25) -> dict[str, float]:
    inv = {pair: 1.0 / max(float(vol), 1e-9) for pair, vol in volatilities.items() if float(vol) > 0}
    total = sum(inv.values()) or 1.0
    weights = {pair: min(value / total, max_weight) for pair, value in inv.items()}
    norm = sum(weights.values()) or 1.0
    return {pair: value / norm for pair, value in weights.items()}


# ── Almgren-Chriss Optimal Execution & Market Impact Model ──────────────────

def almgren_chriss_impact(order_size: float, adv: float, volatility: float,
                         risk_aversion: float = 1e-6, eta_coeff: float = 0.14) -> dict:
    """Almgren-Chriss (2001) Optimal Execution & Square-Root Law Market Impact.

    Temporary Impact (bps) ≈ eta * (OrderSize / ADV)^0.6 * 10000
    Permanent Impact (bps) ≈ 0.5 * Temporary Impact
    Optimal execution horizon T* and child order count N under risk-averse execution.
    """
    size = max(0.0, float(order_size))
    daily_vol = max(1e-12, float(adv))
    vol = max(1e-6, float(volatility))
    gamma = max(1e-9, float(risk_aversion))

    participation_rate = min(1.0, size / daily_vol)
    # Square-root law market impact
    temp_impact_bps = float(eta_coeff) * (participation_rate ** 0.6) * 10000.0
    perm_impact_bps = 0.5 * temp_impact_bps
    total_impact_bps = temp_impact_bps + perm_impact_bps

    # Almgren-Chriss optimal half-life of execution tau = sqrt(eta / (lambda * sigma^2))
    kappa = math.sqrt(max(1e-12, gamma * (vol ** 2) / max(eta_coeff, 1e-6)))
    optimal_time_hours = max(0.1, min(24.0, 1.0 / kappa)) if kappa > 0 else 1.0
    recommended_child_orders = max(1, min(50, int(round(optimal_time_hours * 6))))

    return {
        "participation_rate": round(participation_rate, 4),
        "temporary_impact_bps": round(temp_impact_bps, 2),
        "permanent_impact_bps": round(perm_impact_bps, 2),
        "total_impact_bps": round(total_impact_bps, 2),
        "optimal_duration_hours": round(optimal_time_hours, 2),
        "recommended_child_orders": recommended_child_orders,
        "is_safe_notional": total_impact_bps <= 100.0,
    }
