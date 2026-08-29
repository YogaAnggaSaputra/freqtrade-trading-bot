"""Final Supreme Mathematician Integrations.

Pure-Python implementation of:
  1. TF-IDF & Exponential Decay News Sentiment Scorer.
  2. Counterfactual MFE/MAE & Decayed Regret Analyzer.
  3. Kalman State Reconciliation Filter.
  4. Multi-Armed Bandit (Thompson Sampling) Proposal Selector.
  5. NSGA-II Multi-Objective Pareto Fitness Ranker.
  6. Chandelier Exit Volatility Ratchet.
"""
from __future__ import annotations

import math
from statistics import mean


def _finite(values) -> list[float]:
    return [float(v) for v in values if v is not None and math.isfinite(float(v))]


# ── 1. TF-IDF & Exponential Decay News Sentiment ────────────────────────────

def tfidf_decay_sentiment(headline: str, pos_weights: dict[str, float],
                          neg_weights: dict[str, float], elapsed_minutes: float,
                          half_life_min: float = 60.0) -> float:
    """TF-IDF Weighted & Exponentially Decayed News Sentiment Score ∈ [-1.0, 1.0]."""
    words = [w.lower() for w in headline.split() if w.isalnum()]
    if not words:
        return 0.0
    tf = {w: words.count(w) for w in set(words)}
    raw_pos = sum((1.0 + math.log(cnt)) * pos_weights.get(w, 0.0) for w, cnt in tf.items())
    raw_neg = sum((1.0 + math.log(cnt)) * neg_weights.get(w, 0.0) for w, cnt in tf.items())
    raw_score = raw_pos - raw_neg
    decay = math.exp(-math.log(2.0) * max(0.0, float(elapsed_minutes)) / max(0.1, float(half_life_min)))
    return max(-1.0, min(1.0, raw_score * decay))


# ── 2. Counterfactual MFE/MAE & Decayed Exit Regret ─────────────────────────

def counterfactual_exit_regret(prices: list[float], exit_price: float,
                               side: str = "long", decay_gamma: float = 0.05) -> dict:
    """Decompose post-exit returns into MFE, MAE, and Exponentially Decayed Regret Score."""
    p = _finite(prices)
    ep = max(1e-12, float(exit_price))
    if not p:
        return {"mfe": 0.0, "mae": 0.0, "decayed_regret": 0.0, "opportunity_loss": 0.0}

    is_long = side.lower() in ("long", "buy")
    returns = [(v - ep) / ep if is_long else (ep - v) / ep for v in p]

    mfe = max(returns)
    mae = min(returns)

    weights = [math.exp(-float(decay_gamma) * k) for k in range(1, len(returns) + 1)]
    sum_w = sum(weights) or 1.0
    decayed_regret = sum(max(0.0, r) * w for r, w in zip(returns, weights)) / sum_w

    return {
        "mfe": round(mfe, 4),
        "mae": round(mae, 4),
        "decayed_regret": round(decayed_regret, 4),
        "opportunity_loss": round(max(0.0, mfe), 4),
    }


# ── 3. Kalman State Reconciliation Filter ────────────────────────────────────

class KalmanReconciler:
    """Kalman Filter for CEX vs Local DB Equity/Position Drift Reconciliation.

    Distinguishes temporary execution latency from TRUE state drift anomalies (z-score > 3.0).
    """

    def __init__(self, process_var: float = 1e-5, measurement_var: float = 1e-3):
        self.q = max(1e-12, float(process_var))
        self.r = max(1e-12, float(measurement_var))
        self.x = 0.0
        self.p = 1.0

    def update(self, observed_diff: float) -> dict:
        self.p += self.q
        k = self.p / (self.p + self.r)
        diff = float(observed_diff)
        self.x += k * (diff - self.x)
        self.p *= (1.0 - k)
        z_score = abs(diff - self.x) / math.sqrt(self.p + self.r)
        return {
            "filtered_drift": round(self.x, 6),
            "z_score": round(z_score, 2),
            "is_anomaly": z_score > 3.0,
        }


# ── 4. Multi-Armed Bandit (Thompson Sampling) Proposal Selector ─────────────

class ThompsonProposalSelector:
    """Thompson Sampling MAB to pick optimal Hermes agent proposals based on acceptance priors."""

    def __init__(self, proposal_types: list[str]):
        self.counts = {p: {"alpha": 1.0, "beta": 1.0} for p in proposal_types}

    def select(self, seed: int | None = None) -> str:
        import random
        rng = random.Random(seed)
        best_type, best_sample = "", -1.0
        for p, stats in self.counts.items():
            g_a = rng.gammavariate(stats["alpha"], 1.0)
            g_b = rng.gammavariate(stats["beta"], 1.0)
            sample = g_a / (g_a + g_b) if (g_a + g_b) > 0 else 0.5
            if sample > best_sample:
                best_sample, best_type = sample, p
        return best_type or (list(self.counts.keys())[0] if self.counts else "")

    def record_outcome(self, proposal_type: str, success: bool) -> None:
        if proposal_type in self.counts:
            if success:
                self.counts[proposal_type]["alpha"] += 1.0
            else:
                self.counts[proposal_type]["beta"] += 1.0


# ── 5. NSGA-II Multi-Objective Pareto Dominance ─────────────────────────────

def pareto_dominates(obj1: tuple[float, float], obj2: tuple[float, float]) -> bool:
    """Pareto Dominance for (Sharpe [max], MaxDD [min]). Returns True if obj1 dominates obj2."""
    s1, dd1 = obj1
    s2, dd2 = obj2
    return (s1 >= s2 and dd1 <= dd2) and (s1 > s2 or dd1 < dd2)


def pareto_front(trials: list[dict]) -> list[dict]:
    """Filter trials to non-dominated Pareto optimal front (Sharpe vs MaxDD)."""
    pareto = []
    for i, t1 in enumerate(trials):
        obj1 = (float(t1.get("sharpe", 0.0)), float(t1.get("max_drawdown", 1.0)))
        dominated = False
        for j, t2 in enumerate(trials):
            if i != j:
                obj2 = (float(t2.get("sharpe", 0.0)), float(t2.get("max_drawdown", 1.0)))
                if pareto_dominates(obj2, obj1):
                    dominated = True
                    break
        if not dominated:
            pareto.append(t1)
    return pareto


# ── 6. Chandelier Exit Volatility Ratchet ────────────────────────────────────

def chandelier_exit_ratchet(highest_price: float, current_atr: float,
                            previous_stop: float, atr_multiplier: float = 3.0,
                            side: str = "long") -> float:
    """Chandelier Exit Volatility Ratchet. Monotonically trails stoploss."""
    hp = float(highest_price)
    atr = float(current_atr)
    prev = float(previous_stop)
    mult = float(atr_multiplier)
    is_long = side.lower() in ("long", "buy")

    if is_long:
        candidate = hp - (mult * atr)
        return max(prev, candidate)
    else:
        candidate = hp + (mult * atr)
        return min(prev, candidate) if prev > 0 else candidate
