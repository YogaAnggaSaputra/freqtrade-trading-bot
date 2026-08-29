"""Correlation structure: matrices, PCA via cyclic Jacobi eigen-solver, clustering,
and concentration diagnostics. Pure-Python, numerically guarded.
"""
from __future__ import annotations

import math
from statistics import mean


def _finite(values) -> list[float]:
    return [float(v) for v in values if v is not None and math.isfinite(float(v))]


def pearson(xs, ys) -> float:
    x, y = _finite(xs), _finite(ys)
    n = min(len(x), len(y))
    if n < 2:
        return 0.0
    x, y = x[-n:], y[-n:]
    mx, my = mean(x), mean(y)
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    r = cov / den if den > 0 else 0.0
    return max(-1.0, min(1.0, r))


def correlation_matrix(series: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    names = sorted(series)
    return {a: {b: pearson(series[a], series[b]) for b in names} for a in names}


def rolling_correlation(xs, ys, window: int) -> list[float]:
    x, y = _finite(xs), _finite(ys)
    window = max(2, int(window))
    out = []
    for end in range(window, min(len(x), len(y)) + 1):
        out.append(pearson(x[end - window:end], y[end - window:end]))
    return out


def _symmetric(matrix: list[list[float]]) -> list[list[float]]:
    n = len(matrix)
    return [[(float(matrix[i][j]) + float(matrix[j][i])) / 2.0 for j in range(n)] for i in range(n)]


def jacobi_eigen(matrix: list[list[float]], tol: float = 1e-12, max_sweeps: int = 100) -> tuple[list[float], list[list[float]]]:
    """Eigen-decomposition of a real symmetric matrix by cyclic Jacobi rotations.

    Returns (eigenvalues descending, eigenvectors as rows aligned with eigenvalues).
    Eigenvectors are orthonormal; A = V^T diag(λ) V.
    """
    n = len(matrix)
    if n == 0:
        return [], []
    a = _symmetric(matrix)
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(max_sweeps):
        off = math.sqrt(sum(a[i][j] ** 2 for i in range(n) for j in range(n) if i != j))
        if off <= tol:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(a[p][q]) <= 1e-15:
                    continue
                theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
                t = math.copysign(1.0, theta) / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(n):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p] = c * akp - s * akq
                    a[k][q] = s * akp + c * akq
                for k in range(n):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k] = c * apk - s * aqk
                    a[q][k] = s * apk + c * aqk
                for k in range(n):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p] = c * vkp - s * vkq
                    v[k][q] = s * vkp + c * vkq
    pairs = sorted(((a[i][i], [v[k][i] for k in range(n)]) for i in range(n)),
                   key=lambda item: item[0], reverse=True)
    return [round(value, 12) for value, _ in pairs], [vector for _, vector in pairs]


def pca(matrix: list[list[float]]) -> dict:
    """PCA of a symmetric (e.g. correlation/covariance) matrix with concentration stats."""
    eigenvalues, eigenvectors = jacobi_eigen(matrix)
    total = sum(eigenvalues) or 1.0
    ratios = [value / total for value in eigenvalues]
    return {"eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
            "explained_variance_ratio": ratios,
            "effective_rank": effective_number_of_bets(eigenvalues)}


def effective_number_of_bets(eigenvalues: list[float]) -> float:
    """Meucci-style effective breadth: exp(entropy) of normalized eigenvalues.
    n variables perfectly correlated → 1.0; perfectly independent → n."""
    clipped = [max(float(v), 1e-12) for v in eigenvalues if float(v) > 0]
    if not clipped:
        return 0.0
    total = sum(clipped)
    entropy = -sum((v / total) * math.log(v / total) for v in clipped)
    return math.exp(entropy)


def cluster_by_correlation(names: list[str], corr: dict[str, dict[str, float]],
                           threshold: float = 0.80) -> list[list[str]]:
    """Single-linkage agglomerative clustering on distance 1 - ρ.
    Returns clusters (lists of names) whose members chain at correlation >= threshold."""
    distance = {(a, b): 1.0 - max(-1.0, min(1.0, float(corr.get(a, {}).get(b, 0.0))))
                for a in names for b in names if a != b}
    clusters: list[list[str]] = [[name] for name in names]
    while True:
        best = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = min(distance.get((a, b), distance.get((b, a), 0.0))
                        for a in clusters[i] for b in clusters[j])
                if d <= 1.0 - threshold and (best is None or d < best[0]):
                    best = (d, i, j)
        if best is None:
            return clusters
        _, i, j = best
        clusters[i] = clusters[i] + clusters[j]
        clusters.pop(j)


def average_correlation(corr: dict[str, dict[str, float]], names: list[str]) -> float:
    """Mean pairwise correlation within `names` — used as a portfolio concentration guard."""
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    if not pairs:
        return 0.0
    values = [float(corr.get(a, {}).get(b, 0.0)) for a, b in pairs]
    return sum(values) / len(values)
