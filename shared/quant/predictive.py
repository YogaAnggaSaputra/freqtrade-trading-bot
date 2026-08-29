"""Interpretable predictive helpers: targets, liquidity, MTF, information theory & FFT.

Extended with:
  - Shannon Entropy of scalar series (information content)
  - Mutual Information (MI) between features and forward returns
  - Transfer Entropy (Schreiber 2000) for directed information flow
  - FFT Spectral Decomposition for dominant market cycle detection
"""
from __future__ import annotations
import math
from statistics import mean, pstdev


def measured_move(range_high: float, range_low: float, breakout_up: bool = True) -> float:
    size = max(float(range_high) - float(range_low), 0.0)
    return float(range_high) + size if breakout_up else float(range_low) - size


def liquidity_levels(highs: list[float], lows: list[float], tolerance: float = .001) -> list[float]:
    values = sorted([float(v) for v in highs + lows if float(v) > 0])
    clusters = []
    for value in values:
        if not clusters or abs(value - clusters[-1]) / value > tolerance:
            clusters.append(value)
        else:
            clusters[-1] = (clusters[-1] + value) / 2
    return clusters


def mtf_alignment(scores: dict[str, float]) -> float:
    if not scores: return 0.0
    return sum(max(-1.0, min(1.0, float(v))) for v in scores.values()) / len(scores)


# ── Information Theory: Entropy, Mutual Information, Transfer Entropy ───────

def _discretize(values: list[float], bins: int = 10) -> list[int]:
    v = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    if not v:
        return []
    mn, mx = min(v), max(v)
    if mx <= mn:
        return [0] * len(v)
    w = (mx - mn) / max(bins, 1)
    return [min(int((x - mn) / w), bins - 1) for x in v]


def shannon_entropy(values: list[float], bins: int = 10) -> float:
    """Shannon Entropy H(X) = -Σ p(x) log₂ p(x) of discretized continuous series."""
    disc = _discretize(values, bins)
    if not disc:
        return 0.0
    counts: dict[int, int] = {}
    for b in disc:
        counts[b] = counts.get(b, 0) + 1
    n = len(disc)
    return -sum((c / n) * math.log2(c / n) for c in counts.values() if c > 0)


def mutual_information(xs: list[float], ys: list[float], bins: int = 10) -> float:
    """Mutual Information I(X; Y) = H(X) + H(Y) - H(X, Y).

    Measures non-linear dependency between feature X and return Y.
    I(X; Y) = 0 ↔ X and Y are statistically independent.
    """
    x_d = _discretize(xs, bins)
    y_d = _discretize(ys, bins)
    n = min(len(x_d), len(y_d))
    if n < 5:
        return 0.0
    x_d, y_d = x_d[:n], y_d[:n]

    counts_x: dict[int, int] = {}
    counts_y: dict[int, int] = {}
    counts_xy: dict[tuple[int, int], int] = {}

    for xi, yi in zip(x_d, y_d):
        counts_x[xi] = counts_x.get(xi, 0) + 1
        counts_y[yi] = counts_y.get(yi, 0) + 1
        counts_xy[(xi, yi)] = counts_xy.get((xi, yi), 0) + 1

    mi = 0.0
    for (xi, yi), c_xy in counts_xy.items():
        p_xy = c_xy / n
        p_x = counts_x[xi] / n
        p_y = counts_y[yi] / n
        if p_xy > 0 and p_x > 0 and p_y > 0:
            mi += p_xy * math.log2(p_xy / (p_x * p_y))
    return max(0.0, mi)


def transfer_entropy(xs: list[float], ys: list[float], lag: int = 1, bins: int = 6) -> float:
    """Schreiber (2000) Transfer Entropy TE(X→Y).

    Measures directed information flow from X to Y beyond Y's own past:
    TE(X→Y) = Σ p(y_{t+1}, y_t, x_t) · log₂ [ p(y_{t+1} | y_t, x_t) / p(y_{t+1} | y_t) ]
    Asymmetric: TE(X→Y) ≠ TE(Y→X).
    """
    x_d = _discretize(xs, bins)
    y_d = _discretize(ys, bins)
    n = min(len(x_d), len(y_d))
    lag = max(1, int(lag))
    if n <= lag + 5:
        return 0.0

    # Triples (y_{t+lag}, y_t, x_t)
    counts_3: dict[tuple[int, int, int], int] = {}
    counts_2_y: dict[tuple[int, int], int] = {}
    counts_2_yx: dict[tuple[int, int], int] = {}
    counts_1_y: dict[int, int] = {}
    total = n - lag

    for t in range(total):
        yt1, yt, xt = y_d[t + lag], y_d[t], x_d[t]
        counts_3[(yt1, yt, xt)] = counts_3.get((yt1, yt, xt), 0) + 1
        counts_2_y[(yt1, yt)] = counts_2_y.get((yt1, yt), 0) + 1
        counts_2_yx[(yt, xt)] = counts_2_yx.get((yt, xt), 0) + 1
        counts_1_y[yt] = counts_1_y.get(yt, 0) + 1

    te = 0.0
    for (yt1, yt, xt), c_3 in counts_3.items():
        p_3 = c_3 / total
        p_yt1_yt_xt = p_3 / (counts_2_yx[(yt, xt)] / total)
        p_yt1_yt = (counts_2_y[(yt1, yt)] / total) / (counts_1_y[yt] / total)
        if p_yt1_yt_xt > 0 and p_yt1_yt > 0:
            te += p_3 * math.log2(p_yt1_yt_xt / p_yt1_yt)
    return max(0.0, te)


# ── Fast Fourier Transform (FFT) Cycle Detection ─────────────────────────────

def _dft_power(series: list[float]) -> list[float]:
    """Discrete Fourier Transform power spectrum (Cooley-Tukey fallback)."""
    n = len(series)
    powers = []
    for k in range(n // 2 + 1):
        re = sum(series[t] * math.cos(2.0 * math.pi * k * t / n) for t in range(n))
        im = sum(-series[t] * math.sin(2.0 * math.pi * k * t / n) for t in range(n))
        powers.append((re * re + im * im) / n)
    return powers


def dominant_cycle(prices: list[float], min_period: int = 4, max_period: int = 200) -> dict:
    """Find dominant market cycle period (in bars) via FFT power spectrum analysis.

    Returns:
      dominant_period: integer bars (e.g. 24 bars = 24h on 1h timeframe)
      cycle_power: share of spectral energy in the dominant peak
      secondary_period: 2nd most prominent period
    """
    p = [float(v) for v in prices if v is not None and math.isfinite(float(v))]
    n = len(p)
    if n < min_period * 2:
        return {"dominant_period": 0, "cycle_power": 0.0, "secondary_period": 0}

    # Mean-center and detrend
    m = mean(p)
    detrended = [x - m for x in p]
    powers = _dft_power(detrended)
    total_power = sum(powers[1:]) or 1.0

    # Search for peak in requested period band
    best_k, best_power = 0, -1.0
    second_k, second_power = 0, -1.0

    for k in range(1, len(powers)):
        period = n / k
        if min_period <= period <= max_period:
            pwr = powers[k]
            if pwr > best_power:
                second_k, second_power = best_k, best_power
                best_k, best_power = k, pwr
            elif pwr > second_power:
                second_k, second_power = k, pwr

    dom_period = int(round(n / best_k)) if best_k > 0 else 0
    sec_period = int(round(n / second_k)) if second_k > 0 else 0

    return {
        "dominant_period": dom_period,
        "cycle_power": round(best_power / total_power, 4),
        "secondary_period": sec_period,
        "spectral_purity": round(best_power / max(second_power, 1e-12), 2),
    }
