"""Monte Carlo risk engine: path simulation, VaR/CVaR, drawdown-at-risk, ruin probability.

Pure-Python (stdlib only) so it runs inside any service or the freqtrade strategy
process. All randomness is seedable for reproducible risk reports.
"""
from __future__ import annotations

import math
import random
from statistics import mean, pstdev


def _clean(returns: list[float]) -> list[float]:
    return [float(v) for v in returns if v is not None and math.isfinite(float(v))]


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    index = (len(sorted_values) - 1) * max(0.0, min(1.0, q))
    low, high = int(index), min(int(index) + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (index - low)


def value_at_risk(returns: list[float], alpha: float = 0.05) -> float:
    """Historical VaR: the (1−α)-quantile of the loss distribution L = −r
    (positive = loss), so P(L > VaR) ≈ α."""
    vals = sorted(-v for v in _clean(returns))
    return max(0.0, _quantile(vals, 1.0 - alpha))


def conditional_value_at_risk(returns: list[float], alpha: float = 0.05) -> float:
    """Expected Shortfall via the Rockafellar–Uryasev sample estimator
    CVaR = VaR + (1/(αn))·Σ max(Lᵢ − VaR, 0) — a coherent (subadditive) risk
    measure that equals the mean of the worst α tail for clean cuts."""
    vals = [-v for v in _clean(returns)]
    if not vals:
        return 0.0
    var_ = max(0.0, _quantile(sorted(vals), 1.0 - alpha))
    excess = sum(max(loss - var_, 0.0) for loss in vals)
    return max(var_, var_ + excess / max(float(alpha) * len(vals), 1e-12))


def bootstrap_paths(returns: list[float], horizon: int, n_paths: int = 1000,
                    block: int = 1, seed: int | None = None) -> list[list[float]]:
    """Bootstrap future return paths. `block` > 1 uses circular block bootstrap to
    preserve autocorrelation/volatility clustering of the empirical sample."""
    vals = _clean(returns)
    if not vals or horizon < 1 or n_paths < 1:
        return []
    rng = random.Random(seed)
    block = max(1, min(int(block), len(vals)))
    paths: list[list[float]] = []
    for _ in range(int(n_paths)):
        path: list[float] = []
        while len(path) < horizon:
            start = rng.randrange(len(vals))
            for k in range(block):
                if len(path) >= horizon:
                    break
                path.append(vals[(start + k) % len(vals)])
        paths.append(path)
    return paths


def gbm_paths(start_price: float, mu: float, sigma: float, horizon: int,
              n_paths: int = 1000, dt: float = 1.0, seed: int | None = None) -> list[list[float]]:
    """Geometric Brownian Motion paths: dS = mu*S*dt + sigma*S*dW (exact log-Euler)."""
    if start_price <= 0 or horizon < 1 or n_paths < 1:
        return []
    rng = random.Random(seed)
    drift = (float(mu) - 0.5 * float(sigma) ** 2) * float(dt)
    vol = float(sigma) * math.sqrt(float(dt))
    paths = []
    for _ in range(int(n_paths)):
        price, path = float(start_price), []
        for _ in range(int(horizon)):
            price *= math.exp(drift + vol * rng.gauss(0.0, 1.0))
            path.append(price)
        paths.append(path)
    return paths


def equity_paths(returns_paths: list[list[float]], start: float = 1.0) -> list[list[float]]:
    """Compound return paths into multiplicative equity curves."""
    curves = []
    for path in returns_paths:
        equity, curve = float(start), []
        for r in path:
            equity *= 1.0 + max(-1.0, float(r))
            curve.append(equity)
        curves.append(curve)
    return curves


def max_drawdown(equity_curve: list[float]) -> float:
    peak = 0.0
    worst = 0.0
    for value in equity_curve:
        peak = max(peak, float(value))
        if peak > 0:
            worst = max(worst, (peak - float(value)) / peak)
    return worst


def drawdown_at_risk(returns: list[float], horizon: int, alpha: float = 0.05,
                     n_paths: int = 1000, block: int = 1, seed: int | None = None) -> dict:
    """Monte Carlo drawdown-at-risk: worst-`alpha` quantile of max drawdowns over a horizon."""
    curves = equity_paths(bootstrap_paths(returns, horizon, n_paths, block, seed))
    if not curves:
        return {"dar": 0.0, "expected_drawdown": 0.0, "worst": 0.0}
    drawdowns = sorted(max_drawdown(c) for c in curves)
    return {"dar": _quantile(drawdowns, 1.0 - alpha),
            "expected_drawdown": mean(drawdowns),
            "worst": drawdowns[-1]}


def probability_of_ruin(returns: list[float], ruin_level: float = 0.30, horizon: int = 200,
                        n_paths: int = 1000, block: int = 1, seed: int | None = None) -> float:
    """Share of bootstrap paths whose equity ever touches `ruin_level` drawdown from start."""
    curves = equity_paths(bootstrap_paths(returns, horizon, n_paths, block, seed))
    if not curves:
        return 0.0
    ruined = 0
    for curve in curves:
        peak = curve[0] if curve else 1.0
        for value in curve:
            peak = max(peak, value)
            if peak > 0 and (peak - value) / peak >= max(ruin_level, 0.0):
                ruined += 1
                break
    return ruined / len(curves)


def simulate_r_trades(win_rate: float, avg_win_r: float, avg_loss_r: float,
                      n_trades: int = 100, risk_per_trade: float = 0.01,
                      n_paths: int = 1000, seed: int | None = None) -> dict:
    """Trade-level Monte Carlo of an R-multiple edge: equity distribution, risk of ruin
    (touching `ruin_drawdown`), and VaR of final return. Deterministic given `seed`."""
    p = max(0.0, min(1.0, float(win_rate)))
    win_r = max(float(avg_win_r), 0.0)
    loss_r = max(float(avg_loss_r), 1e-9)
    rng = random.Random(seed)
    finals, ruined = [], 0
    for _ in range(int(n_paths)):
        equity, peak = 1.0, 1.0
        max_dd = 0.0
        for _ in range(int(n_trades)):
            equity += risk_per_trade * (win_r if rng.random() < p else -loss_r)
            peak = max(peak, equity)
            max_dd = max(max_dd, (peak - equity) / peak)
        if max_dd >= 0.30:
            ruined += 1
        finals.append(equity - 1.0)
    ordered = sorted(finals)
    return {"expected_return": mean(finals) if finals else 0.0,
            "std": pstdev(finals) if len(finals) > 1 else 0.0,
            "var_5": -_quantile(ordered, 0.05),
            "median_return": _quantile(ordered, 0.50),
            "worst_5pct_return": _quantile(ordered, 0.05),
            "probability_of_ruin_30pct": ruined / len(finals) if finals else 0.0}


def risk_summary(returns: list[float], alpha: float = 0.05) -> dict:
    """One-call tail-risk snapshot: VaR, CVaR, and the CVaR/VaR tail-heaviness ratio."""
    var_ = value_at_risk(returns, alpha)
    cvar = conditional_value_at_risk(returns, alpha)
    return {"var": var_, "cvar": cvar,
            "tail_ratio": cvar / var_ if var_ > 1e-12 else 1.0}
