"""Reusable quantitative utilities shared by strategy and services.

Sub-modules:
  metrics        — Kelly, Sharpe/Calmar, z-score, regime thresholds, factor scoring,
                   Black-Scholes IV + Greeks (Newton-Raphson)
  advanced       — Parkinson vol, autocorrelation, CUSUM, optimal TP,
                   GARCH(1,1) fit/forecast, realized kernel vol,
                   sample entropy, permutation entropy, Lyapunov exponent
  execution      — slippage, order-type choice, volatility parity,
                   Almgren-Chriss optimal execution & market impact
  allocation     — drawdown × regime exposure multiplier
  position       — position health, exit consensus
  position_risk  — funding impact, stress loss, kill-switch levels
  orderbook      — DOM pressure, spoofing, VPIN-like, iceberg evidence
  predictive     — measured move, liquidity levels, MTF alignment,
                   Shannon/Transfer/Mutual entropy, FFT dominant cycle
  montecarlo     — path simulation, VaR/CVaR, drawdown-at-risk, ruin probability
  correlation    — correlation matrices, PCA (Jacobi), clustering, effective bets
  stochastic     — Hurst, Markov chains, Gaussian HMM (Baum-Welch + Viterbi),
                   OU half-life, Kalman filter, fractional differencing,
                   VAR(p) + Granger causality, optimal stopping threshold
  calibration    — Platt scaling, isotonic PAVA, Brier, ECE,
                   Bayesian Beta-Binomial, Thompson Sampling
  portfolio      — covariance, risk parity (CCD), min-variance, component VaR,
                   Brinson-Hood-Beebower attribution, correlation-aware limits
  microstructure — VPIN, Kyle's Lambda, OFI, Roll spread, Amihud illiquidity
  stats_advanced — Mahalanobis, EVT (GPD), KS-test, Wasserstein, Implementation Shortfall
"""

from .metrics import (
    fractional_kelly,
    realized_volatility,
    rolling_sharpe,
    rolling_calmar,
    zscore,
    regime_threshold,
    weighted_factor_score,
    black_scholes_iv,
    black_scholes_price,
)
from .advanced import (
    parkinson_volatility,
    autocorrelation,
    cusum_break,
    optimal_tp,
    garch11_fit,
    garch11_forecast,
    garch11_log_likelihood,
    realized_kernel_vol,
    sample_entropy,
    permutation_entropy,
    lyapunov_exponent,
)
from .execution import (
    slippage_bps,
    choose_order_type,
    volatility_parity_weights,
    almgren_chriss_impact,
)
from .montecarlo import (
    value_at_risk,
    conditional_value_at_risk,
    bootstrap_paths,
    gbm_paths,
    equity_paths,
    max_drawdown,
    drawdown_at_risk,
    probability_of_ruin,
    simulate_r_trades,
    risk_summary,
)
from .correlation import (
    pearson,
    correlation_matrix,
    rolling_correlation,
    jacobi_eigen,
    pca,
    effective_number_of_bets,
    cluster_by_correlation,
    average_correlation,
)
from .stochastic import (
    hurst_exponent,
    transition_matrix,
    matrix_power,
    stationary_distribution,
    regime_forecast,
    GaussianHMM,
    ou_half_life,
    kalman_filter,
    kalman_velocity,
    fractional_diff,
    fracdiff_min_d,
    var_fit,
    optimal_exit_threshold,
)
from .calibration import (
    platt_scale,
    platt_apply,
    isotonic_calibration,
    isotonic_apply,
    brier_score,
    log_loss,
    reliability_bins,
    expected_calibration_error,
    beta_posterior,
    thompson_sample,
)
from .portfolio import (
    covariance_matrix,
    portfolio_volatility,
    risk_parity_weights,
    risk_contributions,
    min_variance_weights,
    correlation_aware_limits,
    diversification_ratio,
    component_var,
    brinson_attribution,
)
from .microstructure import (
    bulk_volume_classification,
    vpin_toxicity,
    kyle_lambda,
    order_flow_imbalance,
    roll_spread,
    amihud_illiquidity,
)
from .predictive import (
    measured_move,
    liquidity_levels,
    mtf_alignment,
    shannon_entropy,
    mutual_information,
    transfer_entropy,
    dominant_cycle,
)
from .stats_advanced import (
    mahalanobis_distance_2d,
    evt_pareto_tail_index,
    kolmogorov_smirnov_2sample,
    wasserstein_distance_1d,
    implementation_shortfall,
)
from .onchain_offchain import (
    whale_netflow_score,
    mvrv_zscore,
    nvt_signal,
    macro_spillover_index,
    recency_weighted_sentiment,
    defiliquidation_cascade_risk,
)
from .supreme_final import (
    tfidf_decay_sentiment,
    counterfactual_exit_regret,
    KalmanReconciler,
    ThompsonProposalSelector,
    pareto_front,
    chandelier_exit_ratchet,
)

__all__ = [
    # metrics
    "fractional_kelly", "realized_volatility", "rolling_sharpe", "rolling_calmar",
    "zscore", "regime_threshold", "weighted_factor_score",
    "black_scholes_iv", "black_scholes_price",
    # advanced
    "parkinson_volatility", "autocorrelation", "cusum_break", "optimal_tp",
    "garch11_fit", "garch11_forecast", "garch11_log_likelihood",
    "realized_kernel_vol", "sample_entropy", "permutation_entropy", "lyapunov_exponent",
    # execution
    "slippage_bps", "choose_order_type", "volatility_parity_weights", "almgren_chriss_impact",
    # montecarlo
    "value_at_risk", "conditional_value_at_risk", "bootstrap_paths", "gbm_paths",
    "equity_paths", "max_drawdown", "drawdown_at_risk", "probability_of_ruin",
    "simulate_r_trades", "risk_summary",
    # correlation
    "pearson", "correlation_matrix", "rolling_correlation", "jacobi_eigen", "pca",
    "effective_number_of_bets", "cluster_by_correlation", "average_correlation",
    # stochastic
    "hurst_exponent", "transition_matrix", "matrix_power", "stationary_distribution",
    "regime_forecast", "GaussianHMM", "ou_half_life",
    "kalman_filter", "kalman_velocity", "fractional_diff", "fracdiff_min_d",
    "var_fit", "optimal_exit_threshold",
    # calibration
    "platt_scale", "platt_apply", "isotonic_calibration", "isotonic_apply",
    "brier_score", "log_loss", "reliability_bins", "expected_calibration_error",
    "beta_posterior", "thompson_sample",
    # portfolio
    "covariance_matrix", "portfolio_volatility", "risk_parity_weights",
    "risk_contributions", "min_variance_weights", "correlation_aware_limits",
    "diversification_ratio", "component_var", "brinson_attribution",
    # microstructure
    "bulk_volume_classification", "vpin_toxicity", "kyle_lambda", "order_flow_imbalance",
    "roll_spread", "amihud_illiquidity",
    # predictive
    "measured_move", "liquidity_levels", "mtf_alignment",
    "shannon_entropy", "mutual_information", "transfer_entropy", "dominant_cycle",
    # stats_advanced
    "mahalanobis_distance_2d", "evt_pareto_tail_index", "kolmogorov_smirnov_2sample",
    "wasserstein_distance_1d", "implementation_shortfall",
    # onchain_offchain
    "whale_netflow_score", "mvrv_zscore", "nvt_signal", "macro_spillover_index",
    "recency_weighted_sentiment", "defiliquidation_cascade_risk",
    # supreme_final
    "tfidf_decay_sentiment", "counterfactual_exit_regret", "KalmanReconciler",
    "ThompsonProposalSelector", "pareto_front", "chandelier_exit_ratchet",
]
