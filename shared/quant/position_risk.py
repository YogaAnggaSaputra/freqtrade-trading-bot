"""Funding impact and conservative position stress calculations."""
from __future__ import annotations

def funding_impact(notional: float, funding_rate: float, periods: int, expected_profit: float = 0) -> dict:
    cost = abs(float(notional)) * abs(float(funding_rate)) * max(int(periods), 0)
    return {"funding_cost": cost, "cost_to_expected_profit": cost / max(abs(float(expected_profit)), 1e-9),
            "exit_recommended": bool(expected_profit and cost >= abs(expected_profit) * .5)}

def stress_loss(notional: float, adverse_move_pct: float, correlation_impact: float = 0) -> dict:
    direct = abs(float(notional)) * abs(float(adverse_move_pct))
    portfolio = direct * (1.0 + max(float(correlation_impact), 0.0))
    return {"direct_loss": direct, "portfolio_loss": portfolio, "loss_pct_notional": portfolio / max(abs(float(notional)), 1e-9)}

def kill_switch_level(daily_drawdown: float, latency_multiplier: float = 1.0, anomaly: bool = False) -> str:
    if anomaly or daily_drawdown >= .10 or latency_multiplier >= 4: return "black"
    if daily_drawdown >= .075 or latency_multiplier >= 3: return "red"
    if daily_drawdown >= .05 or latency_multiplier >= 2: return "orange"
    return "yellow"
