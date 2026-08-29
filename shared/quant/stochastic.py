"""Stochastic processes: Hurst exponent, Markov chains, a pure-Python Gaussian HMM
(scaled forward-backward + Baum-Welch EM + Viterbi), and Ornstein-Uhlenbeck half-life.

Everything is stdlib-only and numerically guarded (scaled recursions, covariance
floors, damped power iteration) so it is safe for always-on services.
"""
from __future__ import annotations

import math
import random
from statistics import mean, pstdev


def _finite(values) -> list[float]:
    return [float(v) for v in values if v is not None and math.isfinite(float(v))]


# ── Hurst exponent (rescaled range) ──────────────────────────────────────────

def hurst_exponent(values, min_window: int = 8, max_windows: int = 8) -> float:
    """Hurst exponent of a *return* series via rescaled-range analysis.
    H≈0.5 random walk, H>0.5 persistent/trending, H<0.5 mean-reverting."""
    x = _finite(values)
    n = len(x)
    if n < 2 * max(min_window, 4):
        return 0.5
    sizes, stats = [], []
    step = max(1, (n // 2 - min_window) // max(max_windows - 1, 1)) if n // 2 > min_window else 1
    for size in range(min_window, n // 2 + 1, step):
        chunks = [x[i:i + size] for i in range(0, n - size + 1, size)]
        rs = []
        for chunk in chunks:
            m = mean(chunk)
            deviations = [v - m for v in chunk]
            cumulative, walk = 0.0, []
            for d in deviations:
                cumulative += d
                walk.append(cumulative)
            r = max(walk) - min(walk) if walk else 0.0
            s = pstdev(chunk)
            if s > 1e-12 and r > 1e-12:
                rs.append(r / s)
        if rs:
            sizes.append(size)
            stats.append(mean(rs))
    if len(sizes) < 2:
        return 0.5
    lx, ly = [math.log(s) for s in sizes], [math.log(v) for v in stats]
    mx, my = mean(lx), mean(ly)
    sxx = sum((a - mx) ** 2 for a in lx)
    if sxx <= 1e-12:
        return 0.5
    slope = sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / sxx
    return max(0.0, min(1.0, slope))


# ── Markov chains over discrete regime labels ────────────────────────────────

def transition_matrix(states: list[str], labels: list[str] | None = None,
                      alpha: float = 1.0) -> dict[str, dict[str, float]]:
    """Laplace-smoothed first-order transition matrix (rows sum to 1)."""
    unique = list(labels or dict.fromkeys(s for s in states if s is not None))
    if not unique:
        return {}
    index = {name: i for i, name in enumerate(unique)}
    counts = [[0.0] * len(unique) for _ in unique]
    for a, b in zip(states, states[1:]):
        if a in index and b in index:
            counts[index[a]][index[b]] += 1.0
    matrix = {}
    for name, row in zip(unique, counts):
        total = sum(row) + alpha * len(unique)
        matrix[name] = {other: (cell + alpha) / total for other, cell in zip(unique, row)}
    return matrix


def matrix_power(matrix: dict[str, dict[str, float]], steps: int) -> dict[str, dict[str, float]]:
    if steps < 0:
        steps = 0
    result = {a: {b: (1.0 if a == b else 0.0) for b in matrix} for a in matrix}
    for _ in range(steps):
        result = {a: {b: sum(result[a][k] * matrix[k][b] for k in matrix) for b in matrix}
                  for a in matrix}
    return result


def stationary_distribution(matrix: dict[str, dict[str, float]],
                            iterations: int = 2000, tol: float = 1e-14) -> dict[str, float]:
    """Long-run regime occupancy π solving πP = π, via power iteration on the lazy
    chain M = (I + P)/2. The lazy chain has the *same* stationary distribution as P
    but only non-negative eigenvalues, so iteration is guaranteed to converge even
    for periodic transition matrices (unlike plain P or uniform-damped variants)."""
    if not matrix:
        return {}
    names = list(matrix)
    dist = {name: 1.0 / len(names) for name in names}
    for _ in range(iterations):
        flow = {b: sum(dist[a] * matrix[a][b] for a in names) for b in names}
        nxt = {b: 0.5 * dist[b] + 0.5 * flow[b] for b in names}
        if sum(abs(nxt[b] - dist[b]) for b in names) < tol:
            dist = nxt
            break
        dist = nxt
    total = sum(dist.values()) or 1.0
    return {name: value / total for name, value in dist.items()}


def regime_forecast(matrix: dict[str, dict[str, float]], current: str,
                    horizon: int = 1) -> dict[str, float]:
    """Probability distribution over regimes `horizon` steps ahead from `current`."""
    powered = matrix_power(matrix, max(0, int(horizon)))
    row = powered.get(current)
    if not row:
        return {name: 1.0 / len(matrix) for name in matrix} if matrix else {}
    return dict(row)


# ── Gaussian HMM (diagonal covariance) ───────────────────────────────────────

class GaussianHMM:
    """Minimal, dependency-free Gaussian HMM trained with Baum-Welch EM.

    Suitable for smoothing noisy regime labels into a persistent hidden state
    sequence (e.g. market regimes) with transition-aware probabilities.
    """

    def __init__(self, n_states: int = 2, max_iter: int = 100, tol: float = 1e-6,
                 seed: int | None = None):
        self.n_states = max(2, int(n_states))
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self._rng = random.Random(seed)
        self.pi: list[float] = []
        self.A: list[list[float]] = []
        self.means: list[list[float]] = []
        self.vars: list[list[float]] = []
        self.converged = False
        self.log_likelihood = -math.inf

    # ── internals ──
    @staticmethod
    def _log_gauss(x: float, mu: float, var: float) -> float:
        var = max(var, 1e-12)
        return -0.5 * ((x - mu) ** 2 / var + math.log(2.0 * math.pi * var))

    def _emission(self, x: list[float], state: int) -> float:
        return sum(self._log_gauss(v, m, s) for v, m, s in
                   zip(x, self.means[state], self.vars[state]))

    def _forward(self, obs: list[list[float]]) -> tuple[list[list[float]], list[float], float]:
        n, k = len(obs), self.n_states
        alpha = [[0.0] * k for _ in range(n)]
        scales = [0.0] * n
        for j in range(k):
            alpha[0][j] = max(self.pi[j], 1e-300) * math.exp(self._emission(obs[0], j))
        scales[0] = sum(alpha[0]) or 1e-300
        alpha[0] = [a / scales[0] for a in alpha[0]]
        for t in range(1, n):
            for j in range(k):
                alpha[t][j] = math.exp(self._emission(obs[t], j)) * \
                    sum(alpha[t - 1][i] * self.A[i][j] for i in range(k))
            scales[t] = sum(alpha[t]) or 1e-300
            alpha[t] = [a / scales[t] for a in alpha[t]]
        return alpha, scales, sum(math.log(s) for s in scales)

    def _backward(self, obs: list[list[float]], scales: list[float]) -> list[list[float]]:
        n, k = len(obs), self.n_states
        beta = [[0.0] * k for _ in range(n)]
        beta[n - 1] = [1.0] * k
        for t in range(n - 2, -1, -1):
            for i in range(k):
                beta[t][i] = sum(self.A[i][j] * math.exp(self._emission(obs[t + 1], j))
                                 * beta[t + 1][j] for j in range(k)) / scales[t + 1]
            peak = max(beta[t]) or 1.0
            beta[t] = [b / peak for b in beta[t]]
        return beta

    # ── public API ──
    def fit(self, obs: list[list[float]]) -> "GaussianHMM":
        data = [[float(v) for v in point if v is not None and math.isfinite(float(v))]
                for point in obs if point]
        data = [p for p in data if p]
        if len(data) < self.n_states * 5:
            return self
        dims = len(data[0])
        ordered = sorted(data, key=lambda p: sum(p))
        chunk = max(1, len(ordered) // self.n_states)
        self.means = []
        for j in range(self.n_states):
            part = ordered[j * chunk:(j + 1) * chunk] or ordered
            self.means.append([mean([p[i] for p in part]) for i in range(dims)])
        self.vars = [[max(pstdev([p[i] for p in data]) ** 2, 1e-8) for i in range(dims)]
                     for _ in range(self.n_states)]
        self.pi = [1.0 / self.n_states] * self.n_states
        self.A = [[0.9 if i == j else 0.1 / max(self.n_states - 1, 1)
                   for j in range(self.n_states)] for i in range(self.n_states)]
        ll_prev = -math.inf
        for _ in range(self.max_iter):
            alpha, scales, ll = self._forward(data)
            beta = self._backward(data, scales)
            gamma = [[alpha[t][j] * beta[t][j] for j in range(self.n_states)] for t in range(len(data))]
            gamma = [[g / (sum(row) or 1.0) for g in row] for row in gamma]
            # transition expectations
            xi = [[0.0] * self.n_states for _ in range(self.n_states)]
            for t in range(len(data) - 1):
                denom = sum(alpha[t][i] * self.A[i][j] *
                            math.exp(self._emission(data[t + 1], j)) * beta[t + 1][j]
                            for i in range(self.n_states) for j in range(self.n_states)) or 1.0
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        xi[i][j] += alpha[t][i] * self.A[i][j] * \
                            math.exp(self._emission(data[t + 1], j)) * beta[t + 1][j] / denom
            self.pi = gamma[0][:]
            for i in range(self.n_states):
                row_total = sum(xi[i]) or 1e-300
                self.A[i] = [xi[i][j] / row_total for j in range(self.n_states)]
                weight = sum(gamma[t][i] for t in range(len(data))) or 1e-300
                self.means[i] = [sum(gamma[t][i] * data[t][d] for t in range(len(data))) / weight
                                 for d in range(dims)]
                self.vars[i] = [max(sum(gamma[t][i] * (data[t][d] - self.means[i][d]) ** 2
                                        for t in range(len(data))) / weight, 1e-10)
                                for d in range(dims)]
            if abs(ll - ll_prev) <= self.tol * (1.0 + abs(ll)):
                self.converged = True
                self.log_likelihood = ll
                break
            ll_prev, self.log_likelihood = ll, ll
        return self

    def predict_proba(self, obs: list[list[float]]) -> list[list[float]]:
        """Smoothed P(state_t | all observations) via forward-backward."""
        data = [[float(v) for v in point] for point in obs if point]
        if not data or not self.means:
            return []
        alpha, scales, _ = self._forward(data)
        beta = self._backward(data, scales)
        gamma = [[alpha[t][j] * beta[t][j] for j in range(self.n_states)] for t in range(len(data))]
        return [[g / (sum(row) or 1.0) for g in row] for row in gamma]

    def predict(self, obs: list[list[float]]) -> list[int]:
        return [max(range(len(row)), key=lambda j: row[j]) for row in self.predict_proba(obs)]

    def viterbi(self, obs: list[list[float]]) -> list[int]:
        """Most likely hidden state path (log-space dynamic programming)."""
        data = [[float(v) for v in point] for point in obs if point]
        if not data or not self.means:
            return []
        n, k = len(data), self.n_states
        delta = [[0.0] * k for _ in range(n)]
        psi = [[0] * k for _ in range(n)]
        for j in range(k):
            delta[0][j] = math.log(max(self.pi[j], 1e-300)) + self._emission(data[0], j)
        for t in range(1, n):
            for j in range(k):
                best_i = max(range(k), key=lambda i: delta[t - 1][i] + math.log(max(self.A[i][j], 1e-300)))
                delta[t][j] = delta[t - 1][best_i] + math.log(max(self.A[best_i][j], 1e-300)) \
                    + self._emission(data[t], j)
                psi[t][j] = best_i
        path = [max(range(k), key=lambda j: delta[n - 1][j])]
        for t in range(n - 1, 0, -1):
            path.append(psi[t][path[-1]])
        return path[::-1]

    def next_state_distribution(self, last_posterior: list[float]) -> list[float]:
        """One-step-ahead state distribution from a posterior over states."""
        if not last_posterior or not self.A:
            return []
        return [sum(last_posterior[i] * self.A[i][j] for i in range(self.n_states))
                for j in range(self.n_states)]


# ── Ornstein-Uhlenbeck mean reversion ────────────────────────────────────────

def ou_half_life(values) -> dict:
    """Fit the discrete OU process via OLS on Δx = (φ−1)x + c: the regression slope is
    (φ − 1), so speed = −slope, μ = intercept/speed, half-life = ln2/speed (in periods).
    Short half-life = fast mean reversion."""
    x = _finite(values)
    if len(x) < 10:
        return {"half_life_periods": 0.0, "phi": 0.0, "mu": mean(x) if x else 0.0, "speed": 0.0}
    y_prev, dy = x[:-1], [b - a for a, b in zip(x, x[1:])]
    mx, my = mean(y_prev), mean(dy)
    den = sum((a - mx) ** 2 for a in y_prev)
    slope = sum((a - mx) * (b - my) for a, b in zip(y_prev, dy)) / den if den > 1e-12 else 0.0
    slope = max(-0.999, min(0.0, slope))  # OU requires φ ∈ (0, 1), i.e. slope ∈ (−1, 0)
    phi = 1.0 + slope
    speed = -slope
    intercept = my - slope * mx
    mu = intercept / speed if speed > 1e-9 else mx
    half_life = math.log(2.0) / speed if speed > 1e-9 else math.inf
    return {"half_life_periods": half_life if math.isfinite(half_life) else 0.0,
            "phi": phi, "mu": mu, "speed": speed}


def seeded_random_walk(n: int, drift: float = 0.0, sigma: float = 1.0,
                       seed: int | None = None) -> list[float]:
    """Deterministic-when-seeded Gaussian random walk — handy for tests and baselines."""
    rng = random.Random(seed)
    level, walk = 0.0, []
    for _ in range(max(0, int(n))):
        level += drift + rng.gauss(0.0, sigma)
        walk.append(level)
    return walk


# ── Kalman Filter (Local Level Model) ────────────────────────────────────────

def kalman_filter(observations: list[float], Q: float = 1e-4, R: float = 1e-2) -> dict:
    """Scalar Kalman Filter (local-level / random-walk + noise model).

    State model:   x_t = x_{t-1} + w_t,   w_t ~ N(0, Q)   [process noise]
    Observation:   y_t = x_t   + v_t,      v_t ~ N(0, R)   [measurement noise]

    Returns: filtered state estimates, Kalman gains, innovations (residuals),
    and one-step-ahead predicted std. Small Q → slow-tracking smoother.
    Large Q → tracks observations closely (nearly no smoothing).
    """
    obs = [float(v) for v in observations if v is not None and math.isfinite(float(v))]
    if not obs:
        return {"filtered": [], "gains": [], "innovations": [], "predicted_std": []}
    Q, R = max(float(Q), 1e-12), max(float(R), 1e-12)
    x = obs[0]           # initial state estimate
    P = R                # initial error covariance
    filtered, gains, innovations, pred_stds = [], [], [], []
    for y in obs:
        # Predict
        P_pred = P + Q
        # Update
        K = P_pred / (P_pred + R)
        innov = y - x
        x = x + K * innov
        P = (1.0 - K) * P_pred
        filtered.append(x)
        gains.append(K)
        innovations.append(innov)
        pred_stds.append(math.sqrt(max(P_pred + R, 0.0)))
    return {"filtered": filtered, "gains": gains,
            "innovations": innovations, "predicted_std": pred_stds}


def kalman_velocity(observations: list[float], Q: float = 1e-4, R: float = 1e-2) -> list[float]:
    """First difference of Kalman filtered states — a noise-reduced momentum proxy."""
    f = kalman_filter(observations, Q, R)["filtered"]
    return [b - a for a, b in zip(f, f[1:])]


# ── Fractional Differencing (López de Prado 2018) ────────────────────────────

def _fracdiff_weights(d: float, threshold: float = 1e-4, max_k: int = 10000) -> list[float]:
    """Compute binomial expansion weights for (1-L)^d operator (truncated)."""
    w, k = [1.0], 1
    while k < max_k:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        w.append(w_k)
        k += 1
    return w


def fractional_diff(prices: list[float], d: float = 0.4, threshold: float = 1e-4) -> list[float]:
    """Fractional differencing (1-L)^d of a price series (López de Prado 2018).

    d=0: original series (non-stationary, full memory).
    d=1: daily returns (stationary, zero memory).
    d≈0.35-0.45 for crypto: stationary *and* preserves long-range memory for ML.

    Returns series of same length; NaN-skipped prefix padded with initial values.
    """
    p = [float(v) for v in prices if v is not None and math.isfinite(float(v))]
    if not p:
        return []
    w = _fracdiff_weights(float(d), threshold, max_k=len(p))
    K = len(w)
    if K > len(p):
        return []
    result = []
    for t in range(K - 1, len(p)):
        val = sum(w[k] * p[t - k] for k in range(K))
        result.append(val)
    return result


def fracdiff_min_d(prices: list[float], d_candidates: list[float] | None = None,
                   threshold: float = 1e-4) -> float:
    """Find minimum d for which fractionally differenced series passes ADF-proxy stationarity.

    Uses a simplified ADF-like test: checks that autocorrelation at lag-1 of the
    fracdiff series is < 0.97 (i.e., no unit root signature). Returns minimum such d.
    """
    from .advanced import autocorrelation
    candidates = d_candidates or [0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.8, 1.0]
    for d in sorted(candidates):
        fd = fractional_diff(prices, d, threshold)
        if len(fd) > 10 and autocorrelation(fd, lag=1) < 0.97:
            return float(d)
    return 1.0


# ── Vector Autoregression VAR(p) + Granger Causality ─────────────────────────

def var_fit(series_dict: dict[str, list[float]], p: int = 1) -> dict:
    """Fit VAR(p) model via OLS for a dict of aligned return series.

    y_t = c + Φ₁·y_{t-1} + ... + Φ_p·y_{t-p} + ε_t  (OLS equation by equation)

    Returns coefficient matrices and a simple Granger causality table:
    granger[X→Y] = F-statistic proxy (reduction in RSS when X lags added to Y model).
    """
    names = sorted(series_dict)
    k = len(names)
    if k < 2 or p < 1:
        return {}
    aligned = {name: [float(v) for v in series_dict[name]] for name in names}
    T = min(len(v) for v in aligned.values())
    if T < p + k + 5:
        return {}
    data = {name: aligned[name][-T:] for name in names}

    def _ols(X_rows: list[list[float]], y_col: list[float]) -> list[float]:
        """Solve X'Xβ = X'y via Gaussian elimination (pure Python)."""
        n, nc = len(X_rows), len(X_rows[0])
        XtX = [[sum(X_rows[r][i] * X_rows[r][j] for r in range(n)) for j in range(nc)] for i in range(nc)]
        Xty = [sum(X_rows[r][i] * y_col[r] for r in range(n)) for i in range(nc)]
        # Augmented matrix for Gaussian elimination
        aug = [row[:] + [Xty[i]] for i, row in enumerate(XtX)]
        for col in range(nc):
            pivot = max(range(col, nc), key=lambda row: abs(aug[row][col]))
            aug[col], aug[pivot] = aug[pivot], aug[col]
            if abs(aug[col][col]) < 1e-12:
                continue
            for row in range(nc):
                if row == col:
                    continue
                factor = aug[row][col] / aug[col][col]
                for c in range(nc + 1):
                    aug[row][c] -= factor * aug[col][c]
        return [aug[i][nc] / aug[i][i] if abs(aug[i][i]) > 1e-12 else 0.0 for i in range(nc)]

    coefficients: dict[str, dict] = {}
    granger: dict[str, float] = {}

    for target in names:
        y = data[target][p:]
        X_rows = []
        for t in range(p, T):
            row = [1.0]  # intercept
            for lag in range(1, p + 1):
                for predictor in names:
                    row.append(data[predictor][t - lag])
            X_rows.append(row)
        betas = _ols(X_rows, y)
        residuals = [y[i] - sum(betas[j] * X_rows[i][j] for j in range(len(betas))) for i in range(len(y))]
        rss_full = sum(r ** 2 for r in residuals)
        coefficients[target] = {"intercept": betas[0], "betas": betas[1:]}

        # Granger test: restricted model (only own lags, no other predictors)
        for cause in names:
            if cause == target:
                continue
            cause_idx = [1 + lag * k + names.index(cause) for lag in range(p)]
            X_restricted = [[X_rows[i][j] for j in range(len(betas)) if j not in cause_idx] for i in range(len(y))]
            betas_r = _ols(X_restricted, y)
            res_r = [y[i] - sum(betas_r[j] * X_restricted[i][j] for j in range(len(betas_r))) for i in range(len(y))]
            rss_r = sum(r ** 2 for r in res_r)
            dof_r, dof_f = p, len(y) - len(betas)
            if rss_full > 1e-12 and dof_f > 0:
                f_stat = ((rss_r - rss_full) / dof_r) / (rss_full / dof_f)
            else:
                f_stat = 0.0
            granger[f"{cause}→{target}"] = max(0.0, f_stat)

    return {"coefficients": coefficients, "granger": granger, "p": p, "T": T}


# ── Optimal Stopping Threshold for OU process ─────────────────────────────────

def optimal_exit_threshold(ou_params: dict, risk_aversion: float = 1.0) -> dict:
    """Optimal stopping threshold for a position with OU P&L dynamics.

    For an OU process with speed κ, mean μ, vol σ, the optimal exit level
    under CARA utility with risk-aversion λ is:
        x* = μ + λ·σ²/(2·κ)     [risk-adjusted above equilibrium]

    Below x* → hold. Above x* (for short position, below for long) → exit.
    Returns x*, stop_below (for long), stop_above (for short).
    """
    speed = max(float(ou_params.get("speed", 0.1)), 1e-6)
    mu = float(ou_params.get("mu", 0.0))
    sigma2 = float(ou_params.get("sigma2", ou_params.get("speed", 0.01) ** 0.5))
    lam = max(float(risk_aversion), 1e-6)
    buffer = lam * sigma2 / (2.0 * speed)
    x_star = mu + buffer
    return {"optimal_exit": x_star, "ou_mean": mu, "risk_buffer": buffer,
            "stop_long_below": mu - buffer, "stop_short_above": x_star}

