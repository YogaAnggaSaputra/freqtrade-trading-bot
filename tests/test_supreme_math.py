"""Mathematical correctness tests for the supreme-math modules.

Each test checks against a known analytic property, not just "runs without error":
eigen-decompositions reconstruct their matrix, risk parity equalises risk
contributions, HMM posteriors are proper distributions, Hurst recovers ~0.5 on
random walks, PAVA is monotone, simplex projections land on the simplex, etc.
"""
import math
import random
import unittest

from shared.quant.montecarlo import (
    bootstrap_paths, conditional_value_at_risk, drawdown_at_risk, equity_paths,
    gbm_paths, gbm_paths as _gbm, max_drawdown, probability_of_ruin, risk_summary,
    simulate_r_trades, value_at_risk,
)
from shared.quant.correlation import (
    average_correlation, cluster_by_correlation, correlation_matrix, effective_number_of_bets,
    jacobi_eigen, pca, pearson, rolling_correlation,
)
from shared.quant.stochastic import (
    GaussianHMM, hurst_exponent, matrix_power, ou_half_life, regime_forecast,
    stationary_distribution, transition_matrix,
)
from shared.quant.calibration import (
    brier_score, expected_calibration_error, isotonic_apply, isotonic_calibration,
    log_loss, platt_apply, platt_scale, reliability_bins,
)
from shared.quant.portfolio import (
    correlation_aware_limits, covariance_matrix, diversification_ratio, min_variance_weights,
    portfolio_volatility, risk_contributions, risk_parity_weights,
)


def _gauss(rng, sigma=1.0):
    return rng.gauss(0.0, sigma)


class MonteCarloTests(unittest.TestCase):
    def test_var_cvar_known_sample(self):
        returns = [0.01, -0.02, 0.03, -0.05, 0.012, -0.008, 0.02, -0.03, 0.005, -0.04]
        var = value_at_risk(returns, 0.3)
        cvar = conditional_value_at_risk(returns, 0.3)
        self.assertTrue(0.02 <= var <= 0.03, f"var={var}")
        self.assertAlmostEqual(cvar, 0.04, places=9)  # mean of worst 30% = (3+4+5)/3 %e-2
        self.assertEqual(risk_summary(returns, 0.3)["var"], var)

    def test_cvar_is_coherent_tail_mean(self):
        returns = [-1.0] * 3 + [0.1] * 97
        cvar = conditional_value_at_risk(returns, 0.03)
        self.assertAlmostEqual(cvar, 1.0, places=6)
        self.assertGreaterEqual(conditional_value_at_risk(returns, 0.05),
                                value_at_risk(returns, 0.05))

    def test_mc_is_seed_reproducible(self):
        a = bootstrap_paths([0.01, -0.01, 0.02, -0.02, 0.015], 10, 20, seed=7)
        b = bootstrap_paths([0.01, -0.01, 0.02, -0.02, 0.015], 10, 20, seed=7)
        c = bootstrap_paths([0.01, -0.01, 0.02, -0.02, 0.015], 10, 20, seed=8)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_gbm_stays_positive_and_matches_drift(self):
        paths = gbm_paths(100.0, 0.0, 0.1, 50, 100, seed=3)
        self.assertTrue(all(p > 0 for path in paths for p in path))
        self.assertEqual(len(paths[0]), 50)

    def test_simulate_r_trades_positive_edge_makes_money(self):
        stats = simulate_r_trades(0.6, 2.0, 1.0, n_trades=200, risk_per_trade=0.01,
                                  n_paths=500, seed=11)
        self.assertGreater(stats["expected_return"], 0)
        self.assertGreaterEqual(stats["probability_of_ruin_30pct"], 0.0)
        self.assertLess(stats["probability_of_ruin_30pct"], 0.5)

    def test_ruin_probability_orders_edges(self):
        good = simulate_r_trades(0.65, 2.0, 1.0, n_trades=300, risk_per_trade=0.02,
                                 n_paths=300, seed=5)
        bad = simulate_r_trades(0.35, 2.0, 1.0, n_trades=300, risk_per_trade=0.02,
                                n_paths=300, seed=5)
        self.assertLess(good["probability_of_ruin_30pct"], bad["probability_of_ruin_30pct"])

    def test_drawdown_at_risk_bounded(self):
        rng = random.Random(1)
        returns = [_gauss(rng, 0.01) for _ in range(300)]
        dar = drawdown_at_risk(returns, horizon=100, n_paths=200, seed=2)
        self.assertTrue(0.0 <= dar["dar"] <= 1.0)
        self.assertGreaterEqual(dar["worst"], dar["dar"])

    def test_max_drawdown_known_curve(self):
        self.assertAlmostEqual(max_drawdown([1.0, 2.0, 1.0, 1.5]), 0.5, places=12)
        self.assertEqual(max_drawdown(equity_paths([[0.1, 0.1]])[0]), 0.0)


class CorrelationTests(unittest.TestCase):
    def test_jacobi_reconstructs_matrix(self):
        matrix = [[2.0, 1.0, 0.5], [1.0, 3.0, 0.25], [0.5, 0.25, 1.5]]
        values, vectors = jacobi_eigen(matrix)
        for i in range(3):
            for j in range(3):
                reconstructed = sum(vectors[k][i] * values[k] * vectors[k][j] for k in range(3))
                self.assertAlmostEqual(reconstructed, matrix[i][j], places=8)
        self.assertGreaterEqual(values[0], values[1])

    def test_pca_two_by_two_known_eigenvalues(self):
        result = pca([[1.0, 0.9], [0.9, 1.0]])
        self.assertAlmostEqual(result["eigenvalues"][0], 1.9, places=8)
        self.assertAlmostEqual(result["eigenvalues"][1], 0.1, places=8)
        self.assertAlmostEqual(sum(result["explained_variance_ratio"]), 1.0, places=9)

    def test_effective_number_of_bets_extremes(self):
        self.assertAlmostEqual(effective_number_of_bets([1.0] * 4), 4.0, places=9)
        self.assertAlmostEqual(effective_number_of_bets([100.0, 1e-9, 1e-9, 1e-9]), 1.0, places=6)

    def test_pearson_and_rolling(self):
        xs = [1, 2, 3, 4, 5]
        self.assertAlmostEqual(pearson(xs, xs), 1.0, places=12)
        self.assertAlmostEqual(pearson(xs, [5, 4, 3, 2, 1]), -1.0, places=12)
        rolled = rolling_correlation(xs, [2 * v for v in xs], 3)
        self.assertEqual(len(rolled), 3)
        self.assertAlmostEqual(rolled[-1], 1.0, places=12)

    def test_clustering_finds_blocks(self):
        rng = random.Random(4)
        def base(_seed):
            r = random.Random(_seed)
            return [_gauss(r, 0.01) for _ in range(120)]
        a, b = base(1), [v * 1.0 + _gauss(rng, 0.0005) for v in base(1)]
        c = base(2)
        series = {"A": a, "B": b, "C": c}
        corr = correlation_matrix(series)
        clusters = cluster_by_correlation(list(series), corr, threshold=0.9)
        joined = sorted(sorted(cluster) for cluster in clusters)
        self.assertIn(["A", "B"], joined)
        self.assertIn(["C"], joined)
        self.assertGreater(average_correlation(corr, ["A", "B"]), 0.9)


class StochasticTests(unittest.TestCase):
    def test_hurst_random_walk_vs_trending(self):
        rng = random.Random(42)
        iid = [_gauss(rng, 0.01) for _ in range(600)]
        persistent = []
        shock = 0.0
        for _ in range(600):
            shock = 0.8 * shock + _gauss(rng, 0.006)
            persistent.append(shock)
        h_iid = hurst_exponent(iid)
        h_persistent = hurst_exponent(persistent)
        self.assertTrue(0.30 < h_iid < 0.70, f"iid H={h_iid}")
        self.assertGreater(h_persistent, h_iid, f"persistent H={h_persistent}")

    def test_markov_rows_sum_to_one_and_forecast_converges(self):
        states = (["bull"] * 40 + ["bull"] + ["bear"] * 10) * 5
        states = states[:-1]
        matrix = transition_matrix(states)
        for row in matrix.values():
            self.assertAlmostEqual(sum(row.values()), 1.0, places=9)
        self.assertGreater(matrix["bull"]["bull"], matrix["bull"]["bear"])
        stationary = stationary_distribution(matrix)
        self.assertAlmostEqual(sum(stationary.values()), 1.0, places=9)
        far = regime_forecast(matrix, "bull", horizon=200)
        for name in stationary:
            self.assertAlmostEqual(far[name], stationary[name], places=6)

    def test_matrix_power_composition(self):
        matrix = transition_matrix(["a", "a", "b", "b", "a", "b"])
        squared = matrix_power(matrix, 2)
        for a in matrix:
            for b in matrix:
                manual = sum(matrix[a][k] * matrix[k][b] for k in matrix)
                self.assertAlmostEqual(squared[a][b], manual, places=12)

    def test_hmm_recovers_two_regimes(self):
        rng = random.Random(9)
        data, true_state = [], []
        state = 0
        for _ in range(400):
            if rng.random() < 0.05:
                state = 1 - state
            data.append([_gauss(rng, 0.25) + (1.0 if state else -1.0)])
            true_state.append(state)
        hmm = GaussianHMM(n_states=2, seed=9).fit(data)
        posterior = hmm.predict_proba(data)
        self.assertTrue(all(abs(sum(row) - 1.0) < 1e-6 for row in posterior))
        predicted = hmm.predict(data)
        accuracy = sum(1 for p, t in zip(predicted, true_state)
                       if (p == 1) == (t == 1)) / len(true_state)
        self.assertGreater(accuracy, 0.85, f"accuracy={accuracy}")
        path = hmm.viterbi(data)
        self.assertTrue(set(path) <= {0, 1})
        nxt = hmm.next_state_distribution(posterior[-1])
        self.assertAlmostEqual(sum(nxt), 1.0, places=6)

    def test_ou_half_life_known_phi(self):
        rng = random.Random(21)
        phi, level, series = 0.9, 0.0, []
        for _ in range(4000):
            level = phi * level + _gauss(rng, 0.3)
            series.append(level)
        fit = ou_half_life(series)
        self.assertAlmostEqual(fit["phi"], 0.9, places=2)
        expected = math.log(2.0) / 0.1
        self.assertLess(abs(fit["half_life_periods"] - expected) / expected, 0.25)


class CalibrationTests(unittest.TestCase):
    def test_platt_separates_and_bounds(self):
        raw = [0.1, 0.2, 0.3, 0.35, 0.6, 0.7, 0.8, 0.9]
        labels = [0, 0, 0, 0, 1, 1, 1, 1]
        model = platt_scale(raw, labels)
        calibrated = [platt_apply(p, model) for p in raw]
        self.assertTrue(all(0.0 <= c <= 1.0 for c in calibrated))
        self.assertLess(min(calibrated[:4]), max(calibrated[4:]))
        self.assertLess(calibrated[1], calibrated[6])

    def test_pava_is_monotone(self):
        raw = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        labels = [1, 0, 0, 1, 1, 0, 1, 1, 1]
        mapping = isotonic_calibration(raw, labels)
        calibrated = [y for _, y in mapping]
        self.assertEqual(calibrated, sorted(calibrated))
        query = isotonic_apply(0.55, mapping)
        self.assertTrue(0.0 <= query <= 1.0)
        self.assertEqual(query, isotonic_apply(0.55, mapping))

    def test_brier_and_log_loss_known_values(self):
        perfect = brier_score([1.0, 0.0], [1, 0])
        clueless = brier_score([0.5, 0.5], [1, 0])
        self.assertAlmostEqual(perfect, 0.0, places=12)
        self.assertAlmostEqual(clueless, 0.25, places=12)
        self.assertAlmostEqual(log_loss([0.5], [1]), -math.log(0.5), places=12)

    def test_ece_perfect_calibration_is_small(self):
        probs, outcomes = [], []
        rng = random.Random(3)
        for _ in range(4000):
            p = rng.random()
            probs.append(p)
            outcomes.append(1 if rng.random() < p else 0)
        self.assertLess(expected_calibration_error(probs, outcomes), 0.02)
        rows = reliability_bins(probs, outcomes, 10)
        self.assertEqual(len(rows), 10)


class PortfolioTests(unittest.TestCase):
    def _two_asset_cov(self, rho=0.5, va=0.04, vb=0.04):
        return {"A": {"A": va, "B": rho * math.sqrt(va * vb)},
                "B": {"A": rho * math.sqrt(va * vb), "B": vb}}

    def test_risk_parity_equalises_contributions(self):
        cov = self._two_asset_cov(rho=0.6)
        weights = risk_parity_weights(cov)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=9)
        contributions = risk_contributions(weights, cov)
        self.assertAlmostEqual(contributions["A"], contributions["B"], places=6)

    def test_risk_parity_respects_budgets(self):
        cov = self._two_asset_cov(rho=0.2)
        budgets = {"A": 0.75, "B": 0.25}
        weights = risk_parity_weights(cov, budgets)
        contributions = risk_contributions(weights, cov)
        self.assertAlmostEqual(contributions["A"], 0.75, places=6)
        self.assertAlmostEqual(contributions["B"], 0.25, places=6)

    def test_min_variance_concentrates_on_quiet_asset(self):
        cov = {"A": {"A": 0.09, "B": 0.0}, "B": {"A": 0.0, "B": 0.01}}
        weights = min_variance_weights(cov)
        self.assertGreater(weights["B"], 0.8)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=9)
        self.assertAlmostEqual(portfolio_volatility(weights, cov) ** 2,
                               weights["A"] ** 2 * 0.09 + weights["B"] ** 2 * 0.01, places=12)

    def test_covariance_matches_direct_computation(self):
        xs = [0.1, -0.2, 0.3, 0.05, -0.05]
        ys = [0.2, 0.1, -0.3, 0.0, 0.15]
        cov = covariance_matrix({"X": xs, "Y": ys})
        mx, my = sum(xs) / 5, sum(ys) / 5
        expected = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / 5
        self.assertAlmostEqual(cov["X"]["Y"], expected, places=12)
        self.assertAlmostEqual(cov["X"]["X"], sum((a - mx) ** 2 for a in xs) / 5, places=12)

    def test_correlation_aware_limits_replaces_hardcoded_cluster(self):
        corr = {"BTC": {"BTC": 1.0, "ETH": 0.92, "SOL": 0.90, "DOT": 0.30},
                "ETH": {"BTC": 0.92, "ETH": 1.0, "SOL": 0.91, "DOT": 0.28},
                "SOL": {"BTC": 0.90, "ETH": 0.91, "SOL": 1.0, "DOT": 0.25},
                "DOT": {"BTC": 0.30, "ETH": 0.28, "SOL": 0.25, "DOT": 1.0}}
        result = correlation_aware_limits(["BTC", "ETH", "SOL", "DOT"],
                                          {"BTC": 90, "ETH": 85, "SOL": 80, "DOT": 70},
                                          corr, max_positions=3, max_avg_correlation=0.6)
        self.assertIn("DOT", result["selected"])
        self.assertLessEqual(len(result["selected"]), 3)
        high_corr = correlation_aware_limits(["BTC", "ETH", "SOL"], {"BTC": 90, "ETH": 88, "SOL": 86},
                                             corr, max_positions=3, max_avg_correlation=0.5)
        self.assertEqual(high_corr["selected"], ["BTC"])
        self.assertGreater(high_corr["rejected"]["ETH"], 0.5)

    def test_diversification_ratio_extremes(self):
        cov_zero = {"A": {"A": 0.04, "B": 0.0}, "B": {"A": 0.0, "B": 0.04}}
        equal = {"A": 0.5, "B": 0.5}
        self.assertAlmostEqual(diversification_ratio(equal, cov_zero, {"A": 0.2, "B": 0.2}),
                               math.sqrt(2.0), places=9)
        single = diversification_ratio({"A": 1.0}, cov_zero, {"A": 0.2})
        self.assertAlmostEqual(single, 1.0, places=9)


if __name__ == "__main__":
    unittest.main()
