"""Probability calibration: turn raw model scores into honest probabilities.

Platt scaling (Newton-Raphson logistic), isotonic regression (PAVA), and
calibration diagnostics (Brier, log-loss, ECE, reliability bins). Stdlib only.
"""
from __future__ import annotations

import math
from statistics import mean


def _pairs(probs, outcomes) -> tuple[list[float], list[float]]:
    p = [float(v) for v in probs if v is not None and math.isfinite(float(v))]
    y = [1.0 if o else 0.0 for o in outcomes][:len(p)]
    p = p[:len(y)]
    return p, y


def _logit(value: float) -> float:
    value = min(max(value, 1e-9), 1.0 - 1e-9)
    return math.log(value / (1.0 - value))


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-min(value, 60.0)))
    e = math.exp(max(value, -60.0))
    return e / (1.0 + e)


def platt_scale(probs, outcomes, max_iter: int = 100, tol: float = 1e-10) -> dict:
    """Fit A*x + B (x = logit of raw probability) to outcomes by Newton-Raphson
    logistic regression. Returns {"a", "b", "converged"}; apply with platt_apply."""
    p, y = _pairs(probs, outcomes)
    if len(p) < 5 or len(set(y)) < 2:
        return {"a": 1.0, "b": 0.0, "converged": False}
    x = [_logit(v) for v in p]
    a, b = 1.0, 0.0
    converged = False
    for _ in range(max_iter):
        grad_a = grad_b = h_aa = h_ab = h_bb = 0.0
        for xi, yi in zip(x, y):
            prob = _sigmoid(a * xi + b)
            grad_a += (prob - yi) * xi
            grad_b += prob - yi
            w = max(prob * (1.0 - prob), 1e-12)
            h_aa += w * xi * xi
            h_ab += w * xi
            h_bb += w
        hessian = [[h_aa, h_ab], [h_ab, h_bb]]
        det = hessian[0][0] * hessian[1][1] - hessian[0][1] ** 2
        if abs(det) < 1e-18:
            break
        step_a = (hessian[1][1] * grad_a - hessian[0][1] * grad_b) / det
        step_b = (hessian[0][0] * grad_b - hessian[1][0] * grad_a) / det
        a, b = a - step_a, b - step_b
        if abs(step_a) < tol and abs(step_b) < tol:
            converged = True
            break
    return {"a": a, "b": b, "converged": converged}


def platt_apply(prob: float, model: dict) -> float:
    """Apply a platt_scale model to a raw probability."""
    a, b = float(model.get("a", 1.0)), float(model.get("b", 0.0))
    return _sigmoid(a * _logit(prob) + b)


def isotonic_calibration(probs, outcomes) -> list[tuple[float, float]]:
    """Fit a monotone calibration map with Pool-Adjacent-Violators.
    Returns sorted (raw_probability, calibrated_probability) breakpoints."""
    p, y = _pairs(probs, outcomes)
    if not p:
        return []
    order = sorted(range(len(p)), key=lambda i: p[i])
    blocks: list[list[list[float]]] = []  # each block: list of outcome values
    for i in order:
        blocks.append([y[i]])
        while len(blocks) >= 2 and mean(blocks[-2]) >= mean(blocks[-1]):
            blocks[-2] = blocks[-2] + blocks[-1]
            blocks.pop()
    calibrated: list[tuple[float, float]] = []
    cursor = 0
    for block in blocks:
        raw = mean(p[order[cursor + k]] for k in range(len(block)))
        calibrated.append((raw, mean(block)))
        cursor += len(block)
    return calibrated


def isotonic_apply(prob: float, mapping: list[tuple[float, float]]) -> float:
    """Evaluate an isotonic_calibration map with linear interpolation and edge clamping."""
    if not mapping:
        return float(prob)
    value = min(max(float(prob), mapping[0][0]), mapping[-1][0])
    for (x0, y0), (x1, y1) in zip(mapping, mapping[1:]):
        if value <= x1:
            if x1 <= x0:
                return y1
            return y0 + (y1 - y0) * (value - x0) / (x1 - x0)
    return mapping[-1][1]


def brier_score(probs, outcomes) -> float:
    """Mean (p − outcome)². 0 = perfect, 0.25 = no-information coin flip."""
    p, y = _pairs(probs, outcomes)
    return sum((pi - yi) ** 2 for pi, yi in zip(p, y)) / len(p) if p else 1.0


def log_loss(probs, outcomes) -> float:
    p, y = _pairs(probs, outcomes)
    if not p:
        return math.inf
    total = 0.0
    for pi, yi in zip(p, y):
        pi = min(max(pi, 1e-12), 1.0 - 1e-12)
        total += -(yi * math.log(pi) + (1.0 - yi) * math.log(1.0 - pi))
    return total / len(p)


def reliability_bins(probs, outcomes, bins: int = 10) -> list[dict]:
    """Per-bin observed rate, mean predicted probability, and count — the
    reliability curve used to spot over/under-confidence."""
    p, y = _pairs(probs, outcomes)
    out = [{"bin_low": i / bins, "bin_high": (i + 1) / bins, "count": 0,
            "mean_predicted": 0.0, "observed_rate": 0.0} for i in range(bins)]
    for pi, yi in zip(p, y):
        idx = min(int(pi * bins), bins - 1)
        out[idx]["count"] += 1
        out[idx]["mean_predicted"] += pi
        out[idx]["observed_rate"] += yi
    for row in out:
        if row["count"]:
            row["mean_predicted"] /= row["count"]
            row["observed_rate"] /= row["count"]
    return out


def expected_calibration_error(probs, outcomes, bins: int = 10) -> float:
    """ECE = Σ (n_b / N) · |mean_p_b − observed_b|. Lower is better; 0 = perfectly calibrated."""
    p, y = _pairs(probs, outcomes)
    if not p:
        return 1.0
    rows = reliability_bins(p, y, bins)
    return sum(row["count"] / len(p) * abs(row["mean_predicted"] - row["observed_rate"])
               for row in rows if row["count"])


# ── Bayesian Beta-Binomial Updating & Thompson Sampling ──────────────────────

def beta_posterior(n_wins: int, n_losses: int, prior_alpha: float = 1.0,
                   prior_beta: float = 1.0) -> dict:
    """Bayesian Beta-Binomial Conjugate Model for Win-Rate Estimation.

    Prior:      Beta(α₀, β₀)   [uninformative default α₀=β₀=1]
    Likelihood: Binomial(wins, losses | p)
    Posterior:  Beta(α₀ + wins, β₀ + losses)

    Returns posterior mean, variance, and 95% credible interval bounds.
    """
    w = max(0, int(n_wins))
    l = max(0, int(n_losses))
    a = float(prior_alpha) + w
    b = float(prior_beta) + l
    total = a + b

    mean_p = a / total
    var_p = (a * b) / (total * total * (total + 1.0))
    std_p = math.sqrt(var_p)

    # Normal approximation to Beta distribution quantiles for credible interval
    z95 = 1.95996
    lower_95 = max(0.0, mean_p - z95 * std_p)
    upper_95 = min(1.0, mean_p + z95 * std_p)

    return {
        "alpha": a,
        "beta": b,
        "posterior_mean": round(mean_p, 4),
        "posterior_std": round(std_p, 4),
        "lower_ci_95": round(lower_95, 4),
        "upper_ci_95": round(upper_95, 4),
        "sample_size": w + l,
    }


def thompson_sample(regime_stats: dict[str, tuple[int, int]], seed: int | None = None) -> str:
    """Thompson Sampling (Multi-Armed Bandit) for Regime / Strategy Selection.

    `regime_stats`: {regime_name: (n_wins, n_losses)}
    Draws a random sample p_r ~ Beta(α_r, β_r) for each regime and selects
    the regime with the highest sampled probability.
    Explores unproven regimes while exploiting proven ones automatically.
    """
    import random
    if not regime_stats:
        return ""
    rng = random.Random(seed)
    best_regime, best_sample = "", -1.0
    for name, (wins, losses) in regime_stats.items():
        a = 1.0 + max(0, wins)
        b = 1.0 + max(0, losses)
        # Beta random variate using Gamma ratio
        g_a = rng.gammavariate(a, 1.0)
        g_b = rng.gammavariate(b, 1.0)
        sample = g_a / (g_a + g_b) if (g_a + g_b) > 0 else 0.5
        if sample > best_sample:
            best_sample, best_regime = sample, name
    return best_regime

