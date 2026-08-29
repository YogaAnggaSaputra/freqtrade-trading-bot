"""Lightweight statistical estimators suitable for an always-on service.

Extended with supreme-math additions:
  - GARCH(1,1) MLE fit + conditional volatility forecast
  - Realized Kernel Volatility (Barndorff-Nielsen & Shephard Bartlett kernel)
  - Sample Entropy (SampEn) — complexity / regularity measure
  - Permutation Entropy — ordinal pattern diversity
  - Lyapunov Exponent — chaos / predictability horizon (Rosenstein algorithm)
"""
from __future__ import annotations
import math
from statistics import mean, pstdev


def parkinson_volatility(highs: list[float], lows: list[float]) -> float:
    terms = [math.log(h / l) ** 2 for h, l in zip(highs, lows) if h > l > 0]
    return math.sqrt(sum(terms) / (4 * len(terms) * math.log(2))) if terms else 0.0


def autocorrelation(values: list[float], lag: int = 1) -> float:
    if len(values) <= lag + 1: return 0.0
    x, y = values[:-lag], values[lag:]
    mx, my = mean(x), mean(y)
    den = math.sqrt(sum((v - mx) ** 2 for v in x) * sum((v - my) ** 2 for v in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / den if den else 0.0


def cusum_break(values: list[float], threshold: float = 5.0) -> dict:
    if len(values) < 10: return {"break": False, "score": 0.0}
    baseline, sigma = mean(values[:-5]), max(pstdev(values[:-5]), 1e-9)
    score = abs(sum((v - baseline) / sigma for v in values[-5:]))
    return {"break": score >= threshold, "score": score}


def optimal_tp(returns_r: list[float], candidates: list[float] | None = None) -> float:
    candidates = candidates or [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    if not returns_r: return 1.5
    return max(candidates, key=lambda tp: sum(min(max(r, -1), tp) for r in returns_r) / len(returns_r))


# ── GARCH(1,1) ───────────────────────────────────────────────────────────────

def garch11_log_likelihood(returns: list[float], omega: float, alpha: float, beta: float) -> float:
    """GARCH(1,1) log-likelihood: L = -½ Σ[log(σ²_t) + ε²_t/σ²_t]."""
    n = len(returns)
    if n < 5:
        return -math.inf
    mu = mean(returns)
    eps2 = [(r - mu) ** 2 for r in returns]
    sigma2 = max(omega / max(1 - alpha - beta, 1e-6), 1e-8)
    ll = 0.0
    for e2 in eps2:
        sigma2 = max(omega + alpha * e2 + beta * sigma2, 1e-10)
        ll -= 0.5 * (math.log(sigma2) + e2 / sigma2)
    return ll


def garch11_fit(returns: list[float], max_iter: int = 200, tol: float = 1e-8) -> dict:
    """Fit GARCH(1,1): σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1} via coordinate ascent MLE.

    Uses grid-search seed + gradient-free coordinate ascent (robust on small samples).
    Returns omega, alpha, beta, long_run_vol, converged.
    """
    from statistics import variance
    r = [float(v) for v in returns if v is not None and math.isfinite(float(v))]
    if len(r) < 20:
        sigma = pstdev(r) if len(r) > 1 else 0.01
        return {"omega": sigma ** 2 * 0.1, "alpha": 0.1, "beta": 0.85,
                "long_run_vol": sigma, "log_likelihood": -math.inf, "converged": False}

    var_r = max(variance(r), 1e-12)
    # Start: Nelson-Siegel-style moments: alpha+beta ≈ 0.94 (crypto persistence)
    alpha, beta = 0.05, 0.89
    omega = var_r * (1.0 - alpha - beta)

    def ll(o, a, b):
        if o <= 0 or a < 0 or b < 0 or a + b >= 1.0:
            return -math.inf
        return garch11_log_likelihood(r, o, a, b)

    best_ll = ll(omega, alpha, beta)
    step = 0.01
    converged = False
    for _ in range(max_iter):
        improved = False
        for param_idx in range(3):
            params = [omega, alpha, beta]
            for sign in (1, -1):
                params[param_idx] += sign * step
                o, a, b = params
                new_ll = ll(o, a, b)
                if new_ll > best_ll + tol:
                    best_ll = new_ll
                    omega, alpha, beta = o, a, b
                    improved = True
                    break
                else:
                    params[param_idx] -= sign * step
        if not improved:
            if step < 1e-7:
                converged = True
                break
            step *= 0.5
    persistence = alpha + beta
    long_run_var = omega / max(1.0 - persistence, 1e-6)
    return {"omega": omega, "alpha": alpha, "beta": beta,
            "long_run_vol": math.sqrt(max(long_run_var, 0.0)),
            "log_likelihood": best_ll, "converged": converged}


def garch11_forecast(returns: list[float], steps: int = 1,
                     omega: float | None = None, alpha: float | None = None,
                     beta: float | None = None) -> dict:
    """Forecast GARCH(1,1) conditional volatility h-steps ahead.

    σ²_{T+h} = σ̄²  + (α+β)^h · (σ²_T − σ̄²)   [mean-reverting GBM variance]
    Returns current conditional vol + h-step forecast + long-run vol.
    """
    if omega is None or alpha is None or beta is None:
        fit = garch11_fit(returns)
        omega, alpha, beta = fit["omega"], fit["alpha"], fit["beta"]
    r = [float(v) for v in returns if v is not None and math.isfinite(float(v))]
    if len(r) < 5:
        return {"sigma_t": 0.0, "sigma_forecast": [0.0] * steps, "long_run_vol": 0.0}
    mu = mean(r)
    sigma2 = max(omega / max(1.0 - alpha - beta, 1e-6), 1e-8)
    for ret in r:
        eps2 = (ret - mu) ** 2
        sigma2 = max(omega + alpha * eps2 + beta * sigma2, 1e-10)
    persistence = alpha + beta
    long_run_var = omega / max(1.0 - persistence, 1e-6)
    forecasts = []
    v = sigma2
    for h in range(1, max(1, int(steps)) + 1):
        v = long_run_var + persistence ** h * (sigma2 - long_run_var)
        forecasts.append(math.sqrt(max(v, 0.0)))
    return {"sigma_t": math.sqrt(max(sigma2, 0.0)),
            "sigma_forecast": forecasts,
            "long_run_vol": math.sqrt(max(long_run_var, 0.0))}


# ── Realized Kernel Volatility (Barndorff-Nielsen & Shephard) ────────────────

def realized_kernel_vol(returns: list[float], bandwidth: int | None = None) -> float:
    """Jump-robust realized kernel volatility using the Bartlett kernel.

    RK = Σ_{h=-H}^{H} k(h/H)·γ_h   where  k(x) = 1 − |x|  (Bartlett)
    γ_h = Σ r_j · r_{j-|h|}           (sample autocovariance of returns)
    H ~ ceil(n^(2/5)) for optimal bias-variance tradeoff.
    """
    r = [float(v) for v in returns if v is not None and math.isfinite(float(v))]
    n = len(r)
    if n < 3:
        return 0.0
    H = int(bandwidth) if bandwidth else max(1, round(n ** 0.4))
    # Autocovariances γ_h for h = 0, 1, ..., H
    gamma = []
    for h in range(H + 1):
        cov = sum(r[j] * r[j - h] for j in range(h, n)) / n
        gamma.append(cov)
    # Bartlett kernel: k(x) = 1 - |x|
    rk = gamma[0] + 2.0 * sum((1.0 - h / (H + 1)) * gamma[h] for h in range(1, H + 1))
    return math.sqrt(max(0.0, rk))


# ── Sample Entropy (SampEn) ───────────────────────────────────────────────────

def sample_entropy(values: list[float], m: int = 2, r_factor: float = 0.2) -> float:
    """Sample Entropy SampEn(m, r, N).

    SampEn = −ln(A/B)
      B = count of template pair matches at length m   (Chebyshev distance < r)
      A = count of template pair matches at length m+1
    r = r_factor * σ (similarity threshold).
    High SampEn → complex/irregular/choppy market. Low SampEn → regular/trending.
    """
    x = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    n = len(x)
    if n < m + 2:
        return 0.0
    r = r_factor * max(pstdev(x), 1e-12)

    def _count_matches(length: int) -> int:
        count = 0
        for i in range(n - length):
            for j in range(i + 1, n - length):
                if all(abs(x[i + k] - x[j + k]) < r for k in range(length)):
                    count += 1
        return count

    B = _count_matches(m)
    A = _count_matches(m + 1)
    if B == 0 or A == 0:
        return 0.0
    return -math.log(A / B)


# ── Permutation Entropy ───────────────────────────────────────────────────────

def permutation_entropy(values: list[float], m: int = 3) -> float:
    """Normalized Permutation Entropy PE ∈ [0, 1].

    PE(m) = −Σ p(π)·ln p(π) / ln(m!)
    where π are ordinal rank patterns of length m.
    PE ≈ 0 → perfectly ordered (trending). PE ≈ 1 → maximally random (choppy).
    """
    x = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    n = len(x)
    if n < m + 1:
        return 1.0
    from itertools import permutations as _perms

    def _rank_pattern(segment):
        indexed = sorted(range(m), key=lambda i: segment[i])
        rank = [0] * m
        for r, i in enumerate(indexed):
            rank[i] = r
        return tuple(rank)

    counts: dict[tuple, int] = {}
    for i in range(n - m + 1):
        pat = _rank_pattern(x[i:i + m])
        counts[pat] = counts.get(pat, 0) + 1
    total = sum(counts.values())
    entropy = -sum((c / total) * math.log(c / total) for c in counts.values() if c > 0)
    max_entropy = math.log(math.factorial(m))
    return entropy / max_entropy if max_entropy > 0 else 0.0


# ── Lyapunov Exponent (Rosenstein et al. 1993) ────────────────────────────────

def lyapunov_exponent(values: list[float], m: int = 2, tau: int = 1, max_iter: int = 20) -> float:
    """Largest Lyapunov Exponent λ₁ (Rosenstein, Collins & DeLuca 1993).

    λ₁ > 0 → chaotic / unpredictable (high sensitive dependence).
    λ₁ ≈ 0 → quasi-periodic / marginally stable.
    λ₁ < 0 → stable attractor (strongly mean-reverting).

    Algorithm: embed series in m-dim phase space, track divergence of nearest neighbours.
    """
    x = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    n = len(x)
    N = n - (m - 1) * tau
    if N < max_iter + 5:
        return 0.0

    # Build m-dim delay-embedding vectors y_i = [x_i, x_{i+tau}, ..., x_{i+(m-1)*tau}]
    y = [[x[i + k * tau] for k in range(m)] for i in range(N)]

    # Mean period estimate (skip neighbours within mean_period of each other)
    mean_period = max(1, n // 20)

    def _dist(a, b):
        return max(abs(a[k] - b[k]) for k in range(m))

    # Find nearest neighbours avoiding temporal correlation
    nn = []
    for i in range(N):
        best_d, best_j = math.inf, -1
        for j in range(N):
            if abs(i - j) <= mean_period:
                continue
            d = _dist(y[i], y[j])
            if d < best_d:
                best_d, best_d = d, d
                best_j = j
        nn.append(best_j)

    # Divergence curve: d_i(t) = ||y_{i+t} - y_{nn_i+t}||
    log_divs = []
    for step in range(1, max_iter + 1):
        vals = []
        for i in range(N - step):
            j = nn[i]
            if j < 0 or j + step >= N:
                continue
            d = _dist(y[i + step], y[j + step])
            if d > 1e-12:
                vals.append(math.log(d))
        if vals:
            log_divs.append(mean(vals))

    if len(log_divs) < 3:
        return 0.0
    # Slope of log-divergence curve = λ₁
    xs = list(range(1, len(log_divs) + 1))
    mx, my = mean(xs), mean(log_divs)
    den = sum((v - mx) ** 2 for v in xs)
    return sum((xi - mx) * (yi - my) for xi, yi in zip(xs, log_divs)) / den if den > 1e-12 else 0.0
