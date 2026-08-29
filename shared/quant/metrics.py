"""Small dependency-light risk and market-statistics functions."""
from __future__ import annotations

import math
from statistics import mean, pstdev
from typing import Iterable


def _values(values: Iterable[float]) -> list[float]:
    return [float(v) for v in values if v is not None and math.isfinite(float(v))]


def fractional_kelly(win_rate: float, avg_win_r: float, avg_loss_r: float,
                    fraction: float = 0.25, cap: float = 0.02) -> float:
    """Return a conservative Kelly risk fraction, bounded for live trading."""
    p = min(max(float(win_rate), 0.0), 1.0)
    q = 1.0 - p
    b = max(float(avg_win_r), 0.0)
    loss = max(float(avg_loss_r), 1e-9)
    edge = (p * b - q * loss) / max(b, 1e-9)
    return min(max(edge * max(float(fraction), 0.0), 0.0), max(float(cap), 0.0))


def realized_volatility(returns: Iterable[float], annualization: float = 1.0) -> float:
    vals = _values(returns)
    return pstdev(vals) * math.sqrt(max(float(annualization), 1.0)) if len(vals) > 1 else 0.0


def rolling_sharpe(returns: Iterable[float], risk_free: float = 0.0) -> float:
    vals = _values(returns)
    if len(vals) < 2:
        return 0.0
    sigma = pstdev(vals)
    return (mean(vals) - risk_free) / sigma if sigma else 0.0


def rolling_calmar(returns: Iterable[float]) -> float:
    vals = _values(returns)
    if not vals:
        return 0.0
    equity, peak, drawdown = 1.0, 1.0, 0.0
    for value in vals:
        equity *= 1.0 + value
        peak = max(peak, equity)
        drawdown = max(drawdown, (peak - equity) / peak)
    return (equity ** (1.0 / len(vals)) - 1.0) / drawdown if drawdown else 0.0


def zscore(value: float, history: Iterable[float]) -> float:
    vals = _values(history)
    sigma = pstdev(vals) if len(vals) > 1 else 0.0
    return (float(value) - mean(vals)) / sigma if sigma else 0.0


def regime_threshold(regime: str | None, default: float = 60.0) -> float:
    aliases = {
        "trending_bull": "trending_up", "trending_bear": "trending_down",
        "ranging": "sideways_low_vol",
    }
    key = (regime or "").lower()
    key = aliases.get(key, key)
    return {"trending_up": 60.0, "trending_down": 60.0,
            "breakout": 58.0, "sideways_low_vol": 70.0,
            "sideways_high_vol": 75.0, "choppy": 80.0}.get(key, float(default))


def weighted_factor_score(factors: dict[str, float], weights: dict[str, float] | None = None) -> float:
    """Combine factor values into a bounded 0-100 score.

    Values in [-1, 1] are treated as signed signals and mapped to [0, 100];
    values already expressed as percentages are simply clamped.
    """
    if not factors:
        return 50.0
    weights = weights or {name: 1.0 for name in factors}
    weighted = total = 0.0
    for name, raw in factors.items():
        weight = max(float(weights.get(name, 0.0)), 0.0)
        if not weight:
            continue
        value = float(raw)
        normalized = 50.0 + 50.0 * value if -1.0 <= value <= 1.0 else value
        weighted += max(0.0, min(100.0, normalized)) * weight
        total += weight
    return weighted / total if total else 50.0


# ── Black-Scholes Implied Volatility Solver & Option Greeks (Newton-Raphson) ──

def _norm_cdf(x: float) -> float:
    from .microstructure import _norm_cdf as cdf
    return cdf(x)


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def black_scholes_price(S: float, K: float, T: float, sigma: float, r: float = 0.0, option_type: str = "call") -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(0.0, S - K) if option_type.lower() == "call" else max(0.0, K - S)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type.lower() == "call":
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def black_scholes_iv(market_price: float, S: float, K: float, T: float, r: float = 0.0, option_type: str = "call", max_iter: int = 100, tol: float = 1e-6) -> dict:
    """Newton-Raphson Black-Scholes Implied Volatility solver & Option Greeks.

    Solves C(σ) - C_market = 0 using Vega derivate: σ_{n+1} = σ_n - (C(σ_n) - C_m)/Vega.
    Returns: iv, delta, gamma, vega, theta, converged.
    """
    S, K, T = max(1e-12, float(S)), max(1e-12, float(K)), max(1e-12, float(T))
    mp = max(0.0, float(market_price))
    is_call = option_type.lower() == "call"
    intrinsic = max(0.0, S - K) if is_call else max(0.0, K - S)
    if mp <= intrinsic:
        return {"iv": 0.0, "delta": 1.0 if is_call else -1.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "converged": False}

    # Brenner-Subrahmanyam initial guess: σ₀ ≈ √(2π/T) · C/S
    sigma = math.sqrt(2.0 * math.pi / T) * (mp / S)
    sigma = max(0.01, min(5.0, sigma))
    converged = False

    for _ in range(max_iter):
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        price = S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2) if is_call else K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)
        diff = price - mp
        if abs(diff) < tol:
            converged = True
            break
        vega = S * _norm_pdf(d1) * math.sqrt(T)
        if abs(vega) < 1e-12:
            break
        sigma = max(0.001, min(10.0, sigma - diff / vega))

    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    delta = _norm_cdf(d1) if is_call else _norm_cdf(d1) - 1.0
    gamma = _norm_pdf(d1) / (S * sigma * math.sqrt(T))
    vega = S * _norm_pdf(d1) * math.sqrt(T)
    theta = -(S * _norm_pdf(d1) * sigma) / (2.0 * math.sqrt(T)) - r * K * math.exp(-r * T) * (_norm_cdf(d2) if is_call else _norm_cdf(-d2))

    return {
        "iv": round(sigma, 4),
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "vega": round(vega, 4),
        "theta": round(theta, 4),
        "converged": converged,
    }
