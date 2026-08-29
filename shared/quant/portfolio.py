"""Portfolio construction: covariance estimation, equal-risk-contribution weights
(cyclical coordinate descent on the Spinu formulation), minimum-variance weights
(projected gradient on the simplex), and correlation-aware position limits.

Pure-Python so the allocation contract is identical in every service.
"""
from __future__ import annotations

import math


def _finite_series(values) -> list[float]:
    return [float(v) for v in values if v is not None and math.isfinite(float(v))]


def covariance_matrix(series: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    """Sample covariance matrix (population denominator, mean-centred) from aligned series."""
    names = sorted(series)
    data = {name: _finite_series(series[name]) for name in names}
    n = min((len(v) for v in data.values()), default=0)
    if n < 2:
        return {a: {b: 0.0 for b in names} for a in names}
    trimmed = {name: values[-n:] for name, values in data.items()}
    means = {name: sum(values) / n for name, values in trimmed.items()}
    centred = {name: [v - means[name] for v in values] for name, values in trimmed.items()}
    cov: dict[str, dict[str, float]] = {}
    for a in names:
        cov[a] = {}
        for b in names:
            cov[a][b] = sum(x * y for x, y in zip(centred[a], centred[b])) / n
    return cov


def portfolio_volatility(weights: dict[str, float], cov: dict[str, dict[str, float]]) -> float:
    variance = 0.0
    for a, wa in weights.items():
        for b, wb in weights.items():
            variance += wa * wb * float(cov.get(a, {}).get(b, 0.0))
    return math.sqrt(max(variance, 0.0))


def _as_matrix(cov: dict[str, dict[str, float]]) -> tuple[list[str], list[list[float]]]:
    names = sorted(cov)
    return names, [[float(cov[a][b]) for b in names] for a in names]


def risk_parity_weights(cov: dict[str, dict[str, float]], budgets: dict[str, float] | None = None,
                        max_iter: int = 500, tol: float = 1e-10) -> dict[str, float]:
    """Equal (or budgeted) risk contribution weights via cyclical coordinate descent.

    Solves min ½ wᵀΣw − Σ b_i ln w_i whose optimum satisfies w_i(Σw)_i ∝ b_i —
    each position contributes exactly its risk budget to portfolio variance.
    """
    names, matrix = _as_matrix(cov)
    n = len(names)
    if n == 0:
        return {}
    total_budget = sum(max(0.0, float(budgets.get(name, 0.0))) for name in names) if budgets else 0.0
    b = [(budgets.get(name, 0.0) / total_budget if total_budget > 0 else 1.0 / n) for name in names]
    diag = [max(matrix[i][i], 1e-14) for i in range(n)]
    w = [math.sqrt(b[i] / diag[i]) for i in range(n)]
    for _ in range(max_iter):
        delta = 0.0
        for i in range(n):
            cross = sum(matrix[i][j] * w[j] for j in range(n) if j != i)
            new_w = (-cross + math.sqrt(cross * cross + 4.0 * diag[i] * b[i])) / (2.0 * diag[i])
            delta += abs(new_w - w[i])
            w[i] = new_w
        if delta < tol:
            break
    total = sum(w) or 1.0
    return {name: w[i] / total for i, name in enumerate(names)}


def risk_contributions(weights: dict[str, float], cov: dict[str, dict[str, float]]) -> dict[str, float]:
    """Fraction of portfolio variance contributed by each position (sums to 1)."""
    variance = 0.0
    marginal: dict[str, float] = {}
    for a, wa in weights.items():
        row = sum(float(cov.get(a, {}).get(b, 0.0)) * float(weights.get(b, 0.0)) for b in weights)
        marginal[a] = row
        variance += wa * row
    if variance <= 1e-18:
        return {a: 0.0 for a in weights}
    return {a: weights[a] * marginal[a] / variance for a in weights}


def _project_to_simplex(values: list[float]) -> list[float]:
    """Euclidean projection onto {w ≥ 0, Σw = 1} (Duchi et al.)."""
    n = len(values)
    if n == 0:
        return []
    ordered = sorted(values, reverse=True)
    cumulative, rho, theta = 0.0, 0, 0.0
    for i in range(n):
        cumulative += ordered[i]
        candidate = (cumulative - 1.0) / (i + 1)
        if ordered[i] - candidate > 0:
            rho, theta = i + 1, candidate
    if rho == 0:
        return [1.0 / n] * n
    return [max(v - theta, 0.0) for v in values]


def min_variance_weights(cov: dict[str, dict[str, float]], max_iter: int = 5000,
                         tol: float = 1e-12) -> dict[str, float]:
    """Global minimum-variance weights, long-only and fully invested, by projected
    gradient descent with a safe step 1/L (L = max row sum of |Σ|, a Gershgorin-type
    bound on the largest eigenvalue)."""
    names, matrix = _as_matrix(cov)
    n = len(names)
    if n == 0:
        return {}
    lipschitz = max(sum(abs(matrix[i][j]) for j in range(n)) for i in range(n)) or 1.0
    step = 1.0 / lipschitz
    w = [1.0 / n] * n
    for _ in range(max_iter):
        grad = [sum(matrix[i][j] * w[j] for j in range(n)) for i in range(n)]
        updated = _project_to_simplex([w[i] - step * grad[i] for i in range(n)])
        moved = sum(abs(updated[i] - w[i]) for i in range(n))
        w = updated
        if moved < tol:
            break
    total = sum(w) or 1.0
    return {name: w[i] / total for i, name in enumerate(names)}


def correlation_aware_limits(candidates: list[str], scores: dict[str, float],
                             corr: dict[str, dict[str, float]], max_positions: int = 3,
                             max_avg_correlation: float = 0.65) -> dict:
    """Data-driven replacement for hardcoded pair clusters.

    Greedily admits candidates by descending score, rejecting any that would push
    the average pairwise correlation of the selected book above the cap.
    """
    selected: list[str] = []
    rejected: dict[str, float] = {}
    for pair in sorted(candidates, key=lambda p: float(scores.get(p, 0.0)), reverse=True):
        if len(selected) >= max(1, int(max_positions)):
            rejected[pair] = -1.0  # capacity reached
            continue
        trial = selected + [pair]
        values = [float(corr.get(a, {}).get(b, 0.0))
                  for i, a in enumerate(trial) for b in trial[i + 1:]]
        avg = sum(values) / len(values) if values else 0.0
        if avg <= float(max_avg_correlation):
            selected.append(pair)
        else:
            rejected[pair] = avg
    return {"selected": selected, "rejected": rejected}


def diversification_ratio(weights: dict[str, float], cov: dict[str, dict[str, float]],
                          vols: dict[str, float]) -> float:
    """Weighted average volatility ÷ portfolio volatility — the classic diversification
    multiple. 1.0 = no diversification benefit."""
    weighted_vol = sum(float(weights.get(a, 0.0)) * float(vols.get(a, 0.0)) for a in weights)
    portfolio_vol = portfolio_volatility(weights, cov)
    return weighted_vol / portfolio_vol if portfolio_vol > 1e-18 else 1.0


# ── Component VaR & Marginal VaR (Euler Risk Decomposition) ─────────────────

def component_var(weights: dict[str, float], cov: dict[str, dict[str, float]],
                  alpha: float = 0.05, z_score: float = 1.64485) -> dict:
    """Euler Risk Decomposition: Portfolio VaR into Marginal VaR and Component VaR.

    VaR_p = z_α · σ_p
    Marginal VaR_i = z_α · (Σw)_i / σ_p = ∂VaR_p / ∂w_i
    Component VaR_i = w_i · Marginal VaR_i
    Σ Component VaR_i = Portfolio VaR   (Euler's Theorem for homogeneous functions)
    """
    names = sorted(weights)
    if not names:
        return {"portfolio_var": 0.0, "marginal_var": {}, "component_var": {}, "pct_risk": {}}

    p_vol = portfolio_volatility(weights, cov)
    p_var = float(z_score) * p_vol

    marginal_v: dict[str, float] = {}
    component_v: dict[str, float] = {}
    pct_r: dict[str, float] = {}

    for a in names:
        w_a = float(weights.get(a, 0.0))
        # (Σw)_a = sum_b cov[a][b] * w_b
        cov_w_a = sum(float(cov.get(a, {}).get(b, 0.0)) * float(weights.get(b, 0.0)) for b in names)
        mvar = (z_score * cov_w_a / p_vol) if p_vol > 1e-18 else 0.0
        cvar = w_a * mvar
        marginal_v[a] = round(mvar, 6)
        component_v[a] = round(cvar, 6)
        pct_r[a] = round(cvar / p_var, 4) if p_var > 1e-18 else 0.0

    return {
        "portfolio_var": round(p_var, 6),
        "portfolio_volatility": round(p_vol, 6),
        "marginal_var": marginal_v,
        "component_var": component_v,
        "pct_risk_contribution": pct_r,
    }


# ── Brinson-Hood-Beebower Factor Performance Attribution ────────────────────

def brinson_attribution(actual_weights: dict[str, float], bench_weights: dict[str, float],
                         actual_returns: dict[str, float], bench_returns: dict[str, float]) -> dict:
    """Brinson-Hood-Beebower (1986) Return Attribution.

    Decomposes total active return R_p - R_b into:
      1. Allocation Effect:  Σ (w_i - W_i) * (R_i^b - R_b)   [over/underweighting sectors]
      2. Selection Effect:   Σ W_i * (R_i - R_i^b)           [picking better assets]
      3. Interaction Effect: Σ (w_i - W_i) * (R_i - R_i^b)   [combined decision]
    """
    names = sorted(set(actual_weights) | set(bench_weights))
    w = {a: float(actual_weights.get(a, 0.0)) for a in names}
    W = {a: float(bench_weights.get(a, 0.0)) for a in names}
    r = {a: float(actual_returns.get(a, 0.0)) for a in names}
    R = {a: float(bench_returns.get(a, 0.0)) for a in names}

    # Normalize weights to sum to 1
    total_w = sum(w.values()) or 1.0
    total_W = sum(W.values()) or 1.0
    w = {a: v / total_w for a, v in w.items()}
    W = {a: v / total_W for a, v in W.items()}

    R_bench = sum(W[a] * R[a] for a in names)
    R_port = sum(w[a] * r[a] for a in names)

    alloc_eff, select_eff, interact_eff = 0.0, 0.0, 0.0
    per_asset = {}

    for a in names:
        alloc = (w[a] - W[a]) * (R[a] - R_bench)
        select = W[a] * (r[a] - R[a])
        interact = (w[a] - W[a]) * (r[a] - R[a])
        alloc_eff += alloc
        select_eff += select
        interact_eff += interact
        per_asset[a] = {
            "allocation": round(alloc, 6),
            "selection": round(select, 6),
            "interaction": round(interact, 6),
            "total_active": round(alloc + select + interact, 6),
        }

    return {
        "portfolio_return": round(R_port, 6),
        "benchmark_return": round(R_bench, 6),
        "active_return": round(R_port - R_bench, 6),
        "allocation_effect": round(alloc_eff, 6),
        "selection_effect": round(select_eff, 6),
        "interaction_effect": round(interact_eff, 6),
        "by_asset": per_asset,
    }
