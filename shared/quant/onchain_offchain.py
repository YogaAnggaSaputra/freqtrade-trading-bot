"""On-Chain & Off-Chain Quantitative Analysis Engine.

Pure-Python implementation of:
  1. Whale Netflow & Exchange Flow Imbalance Index.
  2. MVRV (Market Value to Realized Value) & NVT Valuation Scores.
  3. Macro Cross-Market Correlation & Volatility Spillover (DXY / S&P500 / Gold).
  4. Recency-Decayed TF-IDF Sentiment Scorer for Off-Chain News & Social Media.
  5. DeFi Liquidation Cascade Risk Estimator.
"""
from __future__ import annotations

import math
from statistics import mean, pstdev


def _finite(values) -> list[float]:
    return [float(v) for v in values if v is not None and math.isfinite(float(v))]


# ── 1. Whale Netflow & Exchange Flow Imbalance Index ─────────────────────────

def whale_netflow_score(inflow_volume: float, outflow_volume: float, 
                        average_volume: float = 1.0) -> dict:
    """Whale Netflow Score ∈ [-1.0, 1.0].

    Inflow > Outflow  → Whale deposit ke CEX (potential selling pressure / bearish).
    Outflow > Inflow  → Whale withdrawal ke cold storage (buy/accumulation / bullish).
    """
    in_vol = max(0.0, float(inflow_volume))
    out_vol = max(0.0, float(outflow_volume))
    netflow = in_vol - out_vol
    total_flow = in_vol + out_vol
    avg_vol = max(float(average_volume), 1e-12)

    # Normalized netflow relative to total flow and average volume
    ratio = netflow / total_flow if total_flow > 0 else 0.0
    intensity = min(1.0, abs(netflow) / avg_vol)
    score = max(-1.0, min(1.0, ratio * intensity))

    return {
        "netflow_amount": round(netflow, 2),
        "inflow_volume": round(in_vol, 2),
        "outflow_volume": round(out_vol, 2),
        "netflow_score": round(score, 4),  # positive = bearish (inflow), negative = bullish (outflow)
        "signal": "bearish_deposit" if score > 0.20 else ("bullish_withdrawal" if score < -0.20 else "neutral"),
    }


# ── 2. MVRV & NVT Valuation Scores ──────────────────────────────────────────

def mvrv_zscore(market_cap: float, realized_cap: float, 
                historical_mvrv_std: float = 0.5) -> float:
    """MVRV Z-Score = (Market Cap - Realized Cap) / StdDev(Market Cap).

    MVRV > 3.0 → Extreme Overvalued (Top Market Peak).
    MVRV < 0.8 → Undervalued (Bottom Accumulation Zone).
    """
    mc = max(1e-12, float(market_cap))
    rc = max(1e-12, float(realized_cap))
    std = max(1e-12, float(historical_mvrv_std))
    mvrv = mc / rc
    return (mvrv - 1.0) / std


def nvt_signal(market_cap: float, daily_transacted_volume_90d_sma: float) -> float:
    """NVT Signal = Market Cap / 90-day SMA of Daily On-Chain Transacted Volume.

    High NVT  → Network value outpaces transaction volume (bubble/overvalued).
    Low NVT   → High transaction throughput relative to valuation (undervalued).
    """
    mc = max(1e-12, float(market_cap))
    vol_sma = max(1e-12, float(daily_transacted_volume_90d_sma))
    return mc / vol_sma


# ── 3. Macro Cross-Market Spillover ─────────────────────────────────────────

def macro_spillover_index(crypto_returns: list[float], dxy_returns: list[float],
                          sp500_returns: list[float] | None = None) -> dict:
    """Macro Spillover & Beta to US Dollar Index (DXY) and S&P500.

    Crypto traditionally has inverse correlation with DXY (negative beta)
    and positive correlation with Risk-On Assets (S&P500 / Nasdaq).
    """
    from .correlation import pearson
    cr = _finite(crypto_returns)
    dx = _finite(dxy_returns)
    sp = _finite(sp500_returns) if sp500_returns else []

    n = min(len(cr), len(dx))
    if n < 5:
        return {"dxy_correlation": 0.0, "sp500_correlation": 0.0, "macro_regime": "decoupled"}

    cr, dx = cr[-n:], dx[-n:]
    dxy_corr = pearson(cr, dx)

    sp500_corr = 0.0
    if sp and len(sp) >= 5:
        m = min(n, len(sp))
        sp500_corr = pearson(cr[-m:], sp[-m:])

    # Regime categorization
    if dxy_corr < -0.40 and sp500_corr > 0.40:
        regime = "classic_risk_on"
    elif dxy_corr > 0.40:
        regime = "dxy_stress_decoupling"
    elif sp500_corr < -0.40:
        regime = "inverse_equity_divergence"
    else:
        regime = "decoupled"

    return {
        "dxy_correlation": round(dxy_corr, 4),
        "sp500_correlation": round(sp500_corr, 4),
        "macro_regime": regime,
    }


# ── 4. Recency-Decayed TF-IDF Off-Chain Sentiment ───────────────────────────

def recency_weighted_sentiment(documents: list[dict], half_life_hours: float = 6.0,
                                target_symbol: str = "BTC") -> dict:
    """Compute Recency-Decayed Off-Chain Sentiment from news/social documents.

    Each document format: {"text": str, "timestamp_age_hours": float}
    Decay factor: w(t) = exp(-ln(2) * age / half_life)
    Score = Σ w(t) * doc_score / Σ w(t)
    """
    if not documents:
        return {"sentiment_score": 0.0, "label": "neutral", "decayed_volume": 0.0}

    hl = max(0.5, float(half_life_hours))
    decay_const = math.log(2.0) / hl
    pos_words = {"approval", "approved", "partnership", "listing", "adoption", "etf", "upgrade", "accumulate", "bullish", "inflow"}
    neg_words = {"hack", "exploit", "lawsuit", "ban", "delist", "liquidation", "fraud", "breach", "dump", "bearish", "outflow"}

    weighted_score_sum = 0.0
    weight_sum = 0.0
    sym = target_symbol.lower()

    for doc in documents:
        text = str(doc.get("text", "")).lower()
        age = max(0.0, float(doc.get("timestamp_age_hours", 0.0)))
        w = math.exp(-decay_const * age)

        # Relevance multiplier
        rel = 1.5 if sym in text else 1.0

        pos_count = sum(w_word in text for w_word in pos_words)
        neg_count = sum(w_word in text for w_word in neg_words)

        if pos_count == 0 and neg_count == 0:
            doc_score = 0.0
        else:
            doc_score = (pos_count - neg_count) / (pos_count + neg_count)

        weighted_score_sum += doc_score * w * rel
        weight_sum += w * rel

    final_score = max(-1.0, min(1.0, weighted_score_sum / weight_sum)) if weight_sum > 0 else 0.0

    return {
        "sentiment_score": round(final_score, 4),
        "label": "bullish" if final_score > 0.15 else ("bearish" if final_score < -0.15 else "neutral"),
        "decayed_volume": round(weight_sum, 2),
        "document_count": len(documents),
    }


# ── 5. DeFi Liquidation Cascade Risk Estimator ──────────────────────────────

def defiliquidation_cascade_risk(current_price: float, debt_positions: list[dict]) -> dict:
    """Estimate On-Chain DeFi Liquidation Cascade Risk.

    `debt_positions`: list of {"liquidation_price": float, "collateral_usdt": float}
    Returns cumulative collateral at risk for 5%, 10%, 20% price drops.
    """
    cp = max(1e-12, float(current_price))
    if not debt_positions:
        return {"at_risk_5pct_usdt": 0.0, "at_risk_10pct_usdt": 0.0, "at_risk_20pct_usdt": 0.0, "cascade_threat": "low"}

    risk_5 = sum(float(p.get("collateral_usdt", 0.0)) for p in debt_positions
                 if float(p.get("liquidation_price", 0.0)) >= cp * 0.95)
    risk_10 = sum(float(p.get("collateral_usdt", 0.0)) for p in debt_positions
                  if float(p.get("liquidation_price", 0.0)) >= cp * 0.90)
    risk_20 = sum(float(p.get("collateral_usdt", 0.0)) for p in debt_positions
                  if float(p.get("liquidation_price", 0.0)) >= cp * 0.80)

    threat = "critical" if risk_10 > 50000000 else ("high" if risk_10 > 10000000 else "low")

    return {
        "current_price": cp,
        "at_risk_5pct_usdt": round(risk_5, 2),
        "at_risk_10pct_usdt": round(risk_10, 2),
        "at_risk_20pct_usdt": round(risk_20, 2),
        "cascade_threat": threat,
    }
