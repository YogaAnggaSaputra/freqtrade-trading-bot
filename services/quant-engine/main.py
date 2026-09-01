import math
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from sqlalchemy import text
from shared.db.session import AsyncSessionLocal, init_db, close_db
from shared.quant import (
    regime_threshold, fractional_kelly, parkinson_volatility, autocorrelation,
    cusum_break, optimal_tp, weighted_factor_score,
    # Supreme Math modules
    value_at_risk, conditional_value_at_risk, drawdown_at_risk, probability_of_ruin,
    simulate_r_trades, risk_summary, correlation_matrix, pca, cluster_by_correlation,
    average_correlation, hurst_exponent, transition_matrix, stationary_distribution,
    regime_forecast, GaussianHMM, ou_half_life, platt_scale, platt_apply,
    isotonic_calibration, isotonic_apply, expected_calibration_error,
    brier_score, covariance_matrix, risk_parity_weights, min_variance_weights,
    correlation_aware_limits, diversification_ratio,
)
from shared.schemas import QuantParams

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield
    await close_db()

app = FastAPI(title="Quant Engine", version="2.0.0 (Supreme Math Edition)", lifespan=lifespan)

def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 1.5
    ordered = sorted(values)
    index = (len(ordered) - 1) * max(0.0, min(1.0, quantile))
    low, high = int(index), min(int(index) + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "quant-engine", "edition": "supreme-math"}

@app.get("/params/{pair}", response_model=QuantParams)
async def params(pair: str, regime: str = "unknown"):
    risk = float(os.getenv("QUANT_DEFAULT_RISK_PCT", "0.01"))
    sample_count = 0
    win_rate = avg_win_r = avg_loss_r = 0.0
    tp3_rrr = float(os.getenv("QUANT_TP3_RRR", "4.0"))
    tp1_rrr = float(os.getenv("QUANT_TP1_RRR", "1.5"))
    tp2_rrr = float(os.getenv("QUANT_TP2_RRR", "2.5"))
    try:
        async with AsyncSessionLocal() as db:
            rows = (await db.execute(text("""
              SELECT pnl_pct, actual_rr FROM trade_outcomes
              WHERE pair=:pair AND (:regime='unknown' OR COALESCE(regime_at_entry,'unknown')=:regime)
              ORDER BY timestamp_exit DESC LIMIT 100
            """), {"pair": pair.upper(), "regime": regime})).mappings().all()
        sample_count = len(rows)
        if sample_count >= 10:
            wins = [r for r in rows if float(r["pnl_pct"] or 0) > 0]
            losses = [abs(float(r["actual_rr"] or 0)) for r in rows if float(r["pnl_pct"] or 0) <= 0]
            winning_r = [abs(float(r["actual_rr"] or 0)) for r in wins]
            win_rate = len(wins) / sample_count
            avg_win_r = sum(abs(float(r["actual_rr"] or 0)) for r in wins) / max(len(wins), 1)
            avg_loss_r = sum(losses) / max(len(losses), 1)
            risk = fractional_kelly(win_rate, avg_win_r, avg_loss_r)
            tp3_rrr = float(optimal_tp(winning_r, [2.0, 2.5, 3.0, 3.5, 4.0]))
            tp1_rrr = max(1.0, min(tp3_rrr * .60, _percentile(winning_r, .35)))
            tp2_rrr = max(tp1_rrr + .25, min(tp3_rrr * .85, _percentile(winning_r, .65)))
    except Exception:
        pass
    return QuantParams(pair=pair.upper(), regime=regime,
                       sl_atr_multiplier=float(os.getenv("QUANT_SL_ATR_MULTIPLIER", "2.0")),
                       min_rrr=float(os.getenv("QUANT_MIN_RRR", "1.5")),
                       tp1_rrr=tp1_rrr, tp2_rrr=tp2_rrr,
                       confluence_threshold=regime_threshold(regime), tp3_rrr=tp3_rrr,
                       risk_pct=risk,
                       confidence=min(sample_count / 50.0, 1.0))

@app.post("/kelly", response_model=dict)
async def kelly(payload: dict):
    return {"risk_pct": fractional_kelly(payload.get("win_rate", 0),
             payload.get("avg_win_r", 0), payload.get("avg_loss_r", 1))}

@app.post("/factor-score", response_model=dict)
async def factor_score(payload: dict):
    factors = {str(k): float(v) for k, v in (payload.get("factors") or {}).items()}
    weights = {str(k): float(v) for k, v in (payload.get("weights") or {}).items()}
    return {"score": weighted_factor_score(factors, weights or None), "factors": factors}

@app.post("/volatility", response_model=dict)
async def volatility(payload: dict):
    return {"parkinson": parkinson_volatility(payload.get("highs", []), payload.get("lows", []))}

@app.post("/structure", response_model=dict)
async def structure(payload: dict):
    values = [float(v) for v in payload.get("returns", [])]
    return {
        "autocorrelation": autocorrelation(values),
        "cusum": cusum_break(values),
        "hurst": hurst_exponent(values),
        "ou_fit": ou_half_life(payload.get("prices", values)),
    }

@app.post("/optimal-tp", response_model=dict)
async def optimal_take_profit(payload: dict):
    return {"tp_rr": optimal_tp([float(v) for v in payload.get("returns_r", [])])}

# ── SUPREME MATH ENDPOINTS ───────────────────────────────────────────────────

@app.post("/monte-carlo", response_model=dict)
async def monte_carlo_analysis(payload: dict):
    """Path simulation, tail risk (VaR/CVaR), drawdown-at-risk, and risk of ruin."""
    returns = [float(v) for v in payload.get("returns", [])]
    horizon = int(payload.get("horizon", 100))
    n_paths = int(payload.get("n_paths", 1000))
    alpha = float(payload.get("alpha", 0.05))
    seed = payload.get("seed")
    seed = int(seed) if seed is not None else None

    if payload.get("trade_params"):
        tp = payload["trade_params"]
        return simulate_r_trades(
            win_rate=float(tp.get("win_rate", 0.5)),
            avg_win_r=float(tp.get("avg_win_r", 2.0)),
            avg_loss_r=float(tp.get("avg_loss_r", 1.0)),
            n_trades=horizon,
            risk_per_trade=float(tp.get("risk_per_trade", 0.01)),
            n_paths=n_paths,
            seed=seed,
        )

    return {
        "tail_risk": risk_summary(returns, alpha),
        "drawdown_at_risk": drawdown_at_risk(returns, horizon, alpha, n_paths, seed=seed),
        "ruin_probability_30pct": probability_of_ruin(returns, 0.30, horizon, n_paths, seed=seed),
    }

@app.post("/correlation-analysis", response_model=dict)
async def correlation_analysis(payload: dict):
    """Pairwise correlation matrix, PCA (Jacobi eigen-solver), effective bets, and clusters."""
    series = {str(k): [float(v) for v in vals] for k, vals in (payload.get("series") or {}).items()}
    names = sorted(series)
    corr = correlation_matrix(series)
    cov = covariance_matrix(series)
    cov_matrix = [[cov[a][b] for b in names] for a in names] if names else []
    pca_result = pca(cov_matrix) if cov_matrix else {}
    threshold = float(payload.get("cluster_threshold", 0.80))
    clusters = cluster_by_correlation(names, corr, threshold) if names else []

    return {
        "correlation_matrix": corr,
        "pca": pca_result,
        "clusters": clusters,
        "average_correlation": average_correlation(corr, names) if names else 0.0,
    }

@app.post("/stochastic-regime", response_model=dict)
async def stochastic_regime_analysis(payload: dict):
    """Markov chain transitions + stationary distribution + Gaussian HMM smoothing."""
    states = [str(s) for s in payload.get("states", [])]
    obs = payload.get("observations")  # list of feature vectors for HMM
    horizon = int(payload.get("forecast_horizon", 1))

    out: dict = {}
    if states:
        trans = transition_matrix(states)
        stationary = stationary_distribution(trans)
        curr = states[-1] if states else ""
        fcst = regime_forecast(trans, curr, horizon) if curr else {}
        out["markov"] = {
            "transition_matrix": trans,
            "stationary_distribution": stationary,
            "forecast": fcst,
        }

    if obs and isinstance(obs, list) and len(obs) >= 10:
        hmm = GaussianHMM(n_states=int(payload.get("n_states", 3)), seed=42).fit(obs)
        probas = hmm.predict_proba(obs)
        viterbi_path = hmm.viterbi(obs)
        out["hmm"] = {
            "converged": hmm.converged,
            "log_likelihood": round(hmm.log_likelihood, 2) if math.isfinite(hmm.log_likelihood) else None,
            "viterbi_path": viterbi_path,
            "current_posterior": probas[-1] if probas else [],
            "next_state_distribution": hmm.next_state_distribution(probas[-1]) if probas else [],
        }

    return out

@app.post("/calibrate-probabilities", response_model=dict)
async def calibrate_probabilities(payload: dict):
    """Fit Platt scaling and Isotonic PAVA models; report Brier and ECE calibration metrics."""
    probs = [float(v) for v in payload.get("probs", [])]
    labels = [1 if v else 0 for v in payload.get("labels", [])]
    query = payload.get("query_prob")

    platt_model = platt_scale(probs, labels)
    iso_map = isotonic_calibration(probs, labels)

    brier = brier_score(probs, labels)
    ece = expected_calibration_error(probs, labels)

    res = {
        "brier_score": brier,
        "expected_calibration_error": ece,
        "platt_model": platt_model,
        "isotonic_breakpoints": len(iso_map),
    }

    if query is not None:
        q = float(query)
        res["calibrated_platt"] = platt_apply(q, platt_model)
        res["calibrated_isotonic"] = isotonic_apply(q, iso_map)

    return res

@app.post("/portfolio-optimizer", response_model=dict)
async def portfolio_optimizer(payload: dict):
    """Risk parity (CCD), minimum-variance, and correlation-aware limits."""
    series = {str(k): [float(v) for v in vals] for k, vals in (payload.get("series") or {}).items()}
    candidates = payload.get("candidates") or list(series.keys())
    scores = {str(k): float(v) for k, v in (payload.get("scores") or {}).items()}
    max_pos = int(payload.get("max_positions", 3))
    max_corr = float(payload.get("max_avg_correlation", 0.65))

    cov = covariance_matrix(series) if series else {}
    corr = correlation_matrix(series) if series else {}

    rp_weights = risk_parity_weights(cov) if cov else {}
    mv_weights = min_variance_weights(cov) if cov else {}
    limits = correlation_aware_limits(candidates, scores, corr, max_pos, max_corr) if corr else {}

    vols = {name: math.sqrt(cov[name][name]) for name in cov} if cov else {}
    div_ratio = diversification_ratio(rp_weights, cov, vols) if cov and rp_weights else 1.0

    return {
        "risk_parity_weights": rp_weights,
        "min_variance_weights": mv_weights,
        "correlation_limits": limits,
        "diversification_ratio": div_ratio,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
