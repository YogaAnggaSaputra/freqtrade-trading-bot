"""Advanced Multivariate & Statistical Distribution Math.

Pure-Python implementation of:
  1. Mahalanobis Distance for multivariate anomaly detection (robust matrix inversion).
  2. Extreme Value Theory (EVT): Peak-Over-Threshold (POT) Pareto tail fitting.
  3. Two-Sample Kolmogorov-Smirnov (KS) Statistic for distribution drift detection.
  4. 1-Wasserstein Distance (Earth Mover's Distance) for empirical return drift.
  5. Implementation Shortfall (Perold 1988) for trade execution benchmarking.
"""
from __future__ import annotations

import math
from statistics import mean, pstdev


def _finite(values) -> list[float]:
    return [float(v) for v in values if v is not None and math.isfinite(float(v))]


# ── 1. Mahalanobis Distance ──────────────────────────────────────────────────

def _matrix_inv_2d(m: list[list[float]]) -> list[list[float]]:
    det = m[0][0] * m[1][1] - m[0][1] * m[1][0]
    if abs(det) < 1e-12:
        return [[1.0, 0.0], [0.0, 1.0]]
    return [[m[1][1] / det, -m[0][1] / det], [-m[1][0] / det, m[0][0] / det]]


def mahalanobis_distance_2d(point: tuple[float, float], sample: list[tuple[float, float]]) -> float:
    """Compute Mahalanobis distance D_M(x) of a 2D point (e.g. [return_z, vol_z])
    against a multivariate distribution sample: D_M = sqrt((x-μ)^T Σ^-1 (x-μ)).
    """
    if len(sample) < 5:
        return 0.0
    x0 = [p[0] for p in sample]
    x1 = [p[1] for p in sample]
    m0, m1 = mean(x0), mean(x1)
    c00 = sum((a - m0) ** 2 for a in x0) / len(x0)
    c11 = sum((b - m1) ** 2 for b in x1) / len(x1)
    c01 = sum((a - m0) * (b - m1) for a, b in zip(x0, x1)) / len(x0)
    cov = [[max(c00, 1e-8), c01], [c01, max(c11, 1e-8)]]
    inv = _matrix_inv_2d(cov)
    d0 = point[0] - m0
    d1 = point[1] - m1
    quad = d0 * (inv[0][0] * d0 + inv[0][1] * d1) + d1 * (inv[1][0] * d0 + inv[1][1] * d1)
    return math.sqrt(max(0.0, quad))


# ── 2. Extreme Value Theory (EVT) — Pickands-Balkema-de Haan ───────────────

def evt_pareto_tail_index(returns: list[float], threshold_quantile: float = 0.90) -> dict:
    """Extreme Value Theory: Fit Generalized Pareto Distribution (GPD) to upper tail losses.

    Returns tail index ξ (xi) and scale σ (sigma).
    If ξ > 0: Heavy tail (Student-t / Power law) — market is subject to fat-tail crashes.
    If ξ ≈ 0: Exponential tail (Normal / Light tail).
    """
    losses = sorted(-v for v in _finite(returns) if -v > 0)
    if len(losses) < 20:
        return {"xi": 0.0, "sigma": 0.0, "n_exceedances": 0}
    u = losses[int(len(losses) * max(0.5, min(0.98, threshold_quantile)))]
    exceedances = [y - u for y in losses if y > u]
    if len(exceedances) < 5:
        return {"xi": 0.0, "sigma": 0.0, "n_exceedances": len(exceedances)}

    # Pickands / Hill-like Moment estimator for Generalized Pareto Distribution
    m1 = mean(exceedances)
    m2 = mean(y * y for y in exceedances)
    if m2 <= 0:
        return {"xi": 0.0, "sigma": m1, "n_exceedances": len(exceedances)}
    xi = 0.5 * (1.0 - (m1 * m1) / max(m2 - m1 * m1, 1e-12))
    sigma = max(1e-6, 0.5 * m1 * (1.0 + (m1 * m1) / max(m2 - m1 * m1, 1e-12)))
    return {"xi": max(-0.5, min(1.0, xi)), "sigma": sigma, "n_exceedances": len(exceedances), "threshold": u}


# ── 3. Two-Sample Kolmogorov-Smirnov (KS) Test ─────────────────────────────

def kolmogorov_smirnov_2sample(sample1: list[float], sample2: list[float]) -> dict:
    """Two-Sample KS Test: D = sup_x |F1(x) - F2(x)|.

    Measures whether two data samples (e.g. recent returns vs baseline returns)
    come from the same probability distribution.
    D > 0.30 indicates significant market structural drift.
    """
    s1 = sorted(_finite(sample1))
    s2 = sorted(_finite(sample2))
    n1, n2 = len(s1), len(s2)
    if n1 == 0 or n2 == 0:
        return {"d_statistic": 0.0, "p_value_approx": 1.0, "is_drifted": False}

    i1 = i2 = 0
    d_max = 0.0
    while i1 < n1 and i2 < n2:
        v1, v2 = s1[i1], s2[i2]
        if v1 <= v2:
            i1 += 1
        if v2 <= v1:
            i2 += 1
        cdf1 = i1 / n1
        cdf2 = i2 / n2
        d_max = max(d_max, abs(cdf1 - cdf2))

    # Asymptotic p-value approximation for KS test: Kolmogorov distribution
    en = math.sqrt((n1 * n2) / (n1 + n2))
    lambda_val = (en + 0.12 + 0.11 / en) * d_max
    # Truncated sum for Kolmogorov distribution survival function 2 * Σ (-1)^(k-1) exp(-2 k^2 λ^2)
    p_val = 0.0
    if lambda_val > 0:
        p_val = min(1.0, max(0.0, 2.0 * sum((-1) ** (k - 1) * math.exp(-2.0 * (k * lambda_val) ** 2)
                                           for k in range(1, 5))))
    return {"d_statistic": round(d_max, 4), "p_value_approx": round(p_val, 4), "is_drifted": d_max >= 0.25 or p_val < 0.05}


# ── 4. 1-Wasserstein Distance (Earth Mover's Distance) ──────────────────────

def wasserstein_distance_1d(sample1: list[float], sample2: list[float]) -> float:
    """1-Wasserstein Distance W_1(u, v) = ∫ |U(x) - V(x)| dx.

    Measures the minimum 'work' required to transform sample1 distribution into sample2.
    Unlike KS statistic (which only captures max gap), Wasserstein measures global distribution shift.
    """
    s1 = sorted(_finite(sample1))
    s2 = sorted(_finite(sample2))
    n1, n2 = len(s1), len(s2)
    if n1 == 0 or n2 == 0:
        return 0.0

    all_vals = sorted(set(s1 + s2))
    if len(all_vals) <= 1:
        return 0.0

    i1 = i2 = 0
    w_dist = 0.0
    for val_curr, val_next in zip(all_vals, all_vals[1:]):
        while i1 < n1 and s1[i1] <= val_curr:
            i1 += 1
        while i2 < n2 and s2[i2] <= val_curr:
            i2 += 1
        cdf1 = i1 / n1
        cdf2 = i2 / n2
        w_dist += abs(cdf1 - cdf2) * (val_next - val_curr)
    return max(0.0, w_dist)


# ── 5. Implementation Shortfall (Perold 1988) ────────────────────────────────

def implementation_shortfall(decision_price: float, arrival_price: float,
                             execution_price: float, side: str = "buy",
                             fee_bps: float = 4.0) -> dict:
    """Implementation Shortfall (Perold 1988) decomposition.

    Decomposes execution cost into:
      1. Delay Cost / Market Impact = (ArrivalPrice - DecisionPrice) / DecisionPrice
      2. Execution Slippage = (ExecutionPrice - ArrivalPrice) / ArrivalPrice
      3. Explicit Fees = fee_bps / 10000
      Total Shortfall = Delay Cost + Execution Slippage + Explicit Fees
    """
    p_dec = max(1e-12, float(decision_price))
    p_arr = max(1e-12, float(arrival_price))
    p_exec = max(1e-12, float(execution_price))
    sign = 1.0 if side.lower() in ("buy", "long") else -1.0

    delay_cost_bps = sign * (p_arr - p_dec) / p_dec * 10000.0
    slippage_bps = sign * (p_exec - p_arr) / p_arr * 10000.0
    fee_cost_bps = float(fee_bps)
    total_shortfall_bps = delay_cost_bps + slippage_bps + fee_cost_bps

    return {
        "delay_cost_bps": round(delay_cost_bps, 2),
        "slippage_bps": round(slippage_bps, 2),
        "fee_cost_bps": round(fee_cost_bps, 2),
        "total_shortfall_bps": round(total_shortfall_bps, 2),
        "execution_quality": "excellent" if total_shortfall_bps <= 10 else ("acceptable" if total_shortfall_bps <= 35 else "poor"),
    }
