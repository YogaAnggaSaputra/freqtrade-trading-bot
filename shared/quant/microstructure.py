"""Advanced Microstructure Math: VPIN, Kyle's Lambda, Order Flow Imbalance (OFI).

Pure-Python implementation of high-frequency quantitative finance models.
References:
  - Easley, Lopez de Prado, O'Hara (2012): Flow Toxicity and Liquidity in a High-Frequency World (VPIN).
  - Kyle (1985): Continuous Auctions and Informed Trader (Kyle's Lambda).
  - Cont, Kukanov, Stoikov (2014): The Price Impact of Order Book Events (OFI).
"""
from __future__ import annotations

import math
from statistics import mean


def _erf(x: float) -> float:
    """Handbook of Mathematical Functions formula 7.1.26 approximation for erf(x)."""
    sign = 1 if x >= 0 else -1
    x = abs(x)
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
    return sign * y


def _norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function Φ(x)."""
    return 0.5 * (1.0 + _erf(x / math.sqrt(2.0)))


def bulk_volume_classification(delta_price: float, sigma_price: float, volume: float) -> tuple[float, float]:
    """Bulk Volume Classification (BVC) from Easley et al. (2012).

    Splits total `volume` into buy volume (V_tau^B) and sell volume (V_tau^S) using the CDF:
        V_tau^B = Volume * Φ(ΔP / σ_P)
        V_tau^S = Volume * (1 - Φ(ΔP / σ_P))
    """
    vol = max(0.0, float(volume))
    sig = max(float(sigma_price), 1e-9)
    z = float(delta_price) / sig
    p_buy = max(0.0, min(1.0, _norm_cdf(z)))
    v_buy = vol * p_buy
    v_sell = vol * (1.0 - p_buy)
    return v_buy, v_sell


def vpin_toxicity(volume_buckets: list[tuple[float, float]], bucket_size: float = 1000.0) -> float:
    """Volume-Synchronized Probability of Toxicity (VPIN).

    VPIN = Σ |V_b^B - V_b^S| / (N * BucketSize)
    where each bucket contains equal volume `bucket_size`.
    Returns value in [0, 1]. >0.5 indicates toxic order flow (informed trading).
    """
    if not volume_buckets:
        return 0.0
    total_imbalance = sum(abs(float(v_buy) - float(v_sell)) for v_buy, v_sell in volume_buckets)
    total_volume = sum(float(v_buy) + float(v_sell) for v_buy, v_sell in volume_buckets)
    if total_volume <= 1e-12:
        return 0.0
    return total_imbalance / total_volume


def kyle_lambda(price_changes: list[float], net_volumes: list[float]) -> float:
    """Kyle's Lambda (λ): Price impact coefficient from OLS regression ΔP = λ * NetVol + ε.

    λ measures illiquidity: price change in currency units per unit of net volume traded.
    Higher λ = lower market depth / higher price impact per trade.
    """
    p = [float(v) for v in price_changes if math.isfinite(float(v))]
    v = [float(x) for x in net_volumes if math.isfinite(float(x))][:len(p)]
    p = p[:len(v)]
    if len(p) < 3:
        return 0.0
    mv = mean(v)
    mp = mean(p)
    var_v = sum((x - mv) ** 2 for x in v)
    if var_v <= 1e-12:
        return 0.0
    cov_pv = sum((x - mv) * (y - mp) for x, y in zip(v, p))
    return max(0.0, cov_pv / var_v)


def order_flow_imbalance(bids_top: list[tuple[float, float]], asks_top: list[tuple[float, float]],
                         prev_bids_top: list[tuple[float, float]], prev_asks_top: list[tuple[float, float]]) -> float:
    """Order Flow Imbalance (OFI) from Cont, Kukanov, Stoikov (2014).

    OFI_t = e_t^B - e_t^A
    where e_t^B is bid-side order flow and e_t^A is ask-side order flow at best level.
    Positive OFI → buying pressure; Negative OFI → selling pressure.
    """
    if not bids_top or not asks_top or not prev_bids_top or not prev_asks_top:
        return 0.0
    p_b, v_b = float(bids_top[0][0]), float(bids_top[0][1])
    p_b_prev, v_b_prev = float(prev_bids_top[0][0]), float(prev_bids_top[0][1])
    p_a, v_a = float(asks_top[0][0]), float(asks_top[0][1])
    p_a_prev, v_a_prev = float(prev_asks_top[0][0]), float(prev_asks_top[0][1])

    # Bid flow e_t^B
    if p_b > p_b_prev:
        e_b = v_b
    elif p_b == p_b_prev:
        e_b = v_b - v_b_prev
    else:
        e_b = -v_b_prev

    # Ask flow e_t^A
    if p_a < p_a_prev:
        e_a = v_a
    elif p_a == p_a_prev:
        e_a = v_a - v_a_prev
    else:
        e_a = -v_a_prev

    return e_b - e_a


# ── Roll (1984) Bid-Ask Spread Estimator ─────────────────────────────────────

def roll_spread(prices: list[float]) -> float:
    """Roll (1984) implicit bid-ask spread from serial covariance of price changes.

    s = 2 * sqrt(-Cov(ΔP_t, ΔP_{t-1}))
    Negative Cov = bid-ask bounce → spread estimate.
    If Cov ≥ 0 (momentum dominates): return 0.0 (spread unidentifiable).
    """
    p = [float(v) for v in prices if v is not None and math.isfinite(float(v))]
    if len(p) < 4:
        return 0.0
    dp = [p[i] - p[i - 1] for i in range(1, len(p))]
    n = len(dp) - 1
    if n < 2:
        return 0.0
    cov = sum(dp[i] * dp[i + 1] for i in range(n)) / n
    if cov >= 0:
        return 0.0
    return 2.0 * math.sqrt(-cov)


# ── Amihud (2002) Illiquidity Ratio ──────────────────────────────────────────

def amihud_illiquidity(abs_returns: list[float], dollar_volumes: list[float]) -> float:
    """Amihud (2002) illiquidity ratio: ILLIQ = (1/T) * Σ |R_t| / DVOL_t.

    Measures price impact per unit of dollar volume.
    Higher = less liquid / more price-sensitive to volume.
    Scaled by 1e6 for readability (raw values are tiny in liquid markets).
    """
    pairs = [(abs(float(r)), max(float(v), 1e-6))
             for r, v in zip(abs_returns, dollar_volumes)
             if r is not None and v is not None and math.isfinite(float(r)) and math.isfinite(float(v))]
    if not pairs:
        return 0.0
    return 1e6 * sum(r / v for r, v in pairs) / len(pairs)

