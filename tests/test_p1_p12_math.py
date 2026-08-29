"""Unit tests for the 15 Category A Supreme Mathematician additions."""

import math
import random
import unittest

from shared.quant.advanced import (
    garch11_fit, garch11_forecast, realized_kernel_vol,
    sample_entropy, permutation_entropy, lyapunov_exponent,
)
from shared.quant.stochastic import (
    kalman_filter, kalman_velocity, fractional_diff, fracdiff_min_d,
    var_fit, optimal_exit_threshold,
)
from shared.quant.portfolio import component_var, brinson_attribution
from shared.quant.execution import almgren_chriss_impact
from shared.quant.microstructure import roll_spread, amihud_illiquidity
from shared.quant.predictive import (
    shannon_entropy, mutual_information, transfer_entropy, dominant_cycle,
)
from shared.quant.calibration import beta_posterior, thompson_sample
from shared.quant.metrics import black_scholes_price, black_scholes_iv


class CategoryAMathTests(unittest.TestCase):
    # 1. GARCH(1,1)
    def test_garch11_fit_and_forecast(self):
        rng = random.Random(42)
        returns = [rng.gauss(0, 0.02) for _ in range(100)]
        fit = garch11_fit(returns)
        self.assertIn("omega", fit)
        self.assertIn("alpha", fit)
        self.assertIn("beta", fit)
        forecast = garch11_forecast(returns, steps=5)
        self.assertEqual(len(forecast["sigma_forecast"]), 5)
        self.assertGreater(forecast["sigma_t"], 0)

    # 2. Kalman Filter
    def test_kalman_filter_smooths_noise(self):
        rng = random.Random(10)
        # Stationary mean process with random noise
        true_signal = [50.0] * 50
        noisy = [val + rng.gauss(0, 2.0) for val in true_signal]
        filtered = kalman_filter(noisy, Q=1e-4, R=4.0)["filtered"]
        vel = kalman_velocity(noisy, Q=1e-4, R=4.0)
        self.assertEqual(len(filtered), 50)
        self.assertEqual(len(vel), 49)
        # Filtered output should be closer to true mean than raw noise
        err_raw = sum(abs(n - t) for n, t in zip(noisy, true_signal))
        err_filt = sum(abs(f - t) for f, t in zip(filtered, true_signal))
        self.assertLess(err_filt, err_raw)

    # 3. Almgren-Chriss Impact
    def test_almgren_chriss_impact(self):
        impact = almgren_chriss_impact(order_size=5000, adv=1000000, volatility=0.02)
        self.assertIn("temporary_impact_bps", impact)
        self.assertIn("optimal_duration_hours", impact)
        self.assertTrue(impact["is_safe_notional"])
        huge_impact = almgren_chriss_impact(order_size=500000, adv=1000000, volatility=0.05)
        self.assertFalse(huge_impact["is_safe_notional"])

    # 4. Roll spread & Amihud
    def test_roll_spread_and_amihud(self):
        prices = [100.0, 101.0, 99.5, 101.2, 99.8, 100.5, 99.2]
        r_spread = roll_spread(prices)
        self.assertGreaterEqual(r_spread, 0.0)
        illiquidity = amihud_illiquidity([0.01, 0.02], [10000.0, 20000.0])
        self.assertGreater(illiquidity, 0.0)

    # 5. Component / Marginal VaR
    def test_component_var(self):
        weights = {"BTC": 0.6, "ETH": 0.4}
        cov = {"BTC": {"BTC": 0.04, "ETH": 0.02}, "ETH": {"BTC": 0.02, "ETH": 0.09}}
        c_var = component_var(weights, cov)
        self.assertIn("portfolio_var", c_var)
        self.assertIn("component_var", c_var)
        # Component VaRs sum to portfolio VaR (Euler's theorem)
        sum_cvar = sum(c_var["component_var"].values())
        self.assertAlmostEqual(sum_cvar, c_var["portfolio_var"], places=5)

    # 6. Brinson Performance Attribution
    def test_brinson_attribution(self):
        w_act = {"BTC": 0.7, "ETH": 0.3}
        w_bench = {"BTC": 0.5, "ETH": 0.5}
        r_act = {"BTC": 0.05, "ETH": -0.02}
        r_bench = {"BTC": 0.04, "ETH": -0.01}
        attr = brinson_attribution(w_act, w_bench, r_act, r_bench)
        self.assertIn("allocation_effect", attr)
        self.assertIn("selection_effect", attr)
        self.assertIn("active_return", attr)
        self.assertAlmostEqual(
            attr["active_return"],
            attr["allocation_effect"] + attr["selection_effect"] + attr["interaction_effect"],
            places=5
        )

    # 7. Information Theory (Shannon, MI, Transfer Entropy)
    def test_information_theory(self):
        xs = [float(i) for i in range(100)]
        ys = [val * 2.0 for val in xs]
        h_x = shannon_entropy(xs)
        mi = mutual_information(xs, ys)
        te = transfer_entropy(xs, ys)
        self.assertGreater(h_x, 0.0)
        self.assertGreater(mi, 0.0)
        self.assertGreaterEqual(te, 0.0)

    # 8. FFT Dominant Cycle
    def test_dominant_cycle(self):
        # Generate clean sine wave with period = 20 bars
        prices = [100.0 + 10.0 * math.sin(2.0 * math.pi * t / 20.0) for t in range(200)]
        cycle = dominant_cycle(prices, min_period=4, max_period=50)
        self.assertAlmostEqual(cycle["dominant_period"], 20, delta=2)
        self.assertGreater(cycle["cycle_power"], 0.20)

    # 9. Bayesian Beta-Binomial & Thompson Sampling
    def test_bayesian_beta_binomial(self):
        post = beta_posterior(n_wins=30, n_losses=20)
        self.assertEqual(post["alpha"], 31.0)
        self.assertEqual(post["beta"], 21.0)
        self.assertAlmostEqual(post["posterior_mean"], 31.0 / 52.0, places=4)
        self.assertLess(post["lower_ci_95"], post["posterior_mean"])

        stats = {"regime_bull": (40, 10), "regime_bear": (10, 40)}
        sampled = thompson_sample(stats, seed=42)
        self.assertIn(sampled, stats)

    # 10. Fractional Differencing
    def test_fractional_differencing(self):
        prices = [100.0 + float(i) * 0.5 for i in range(100)]
        fd = fractional_diff(prices, d=0.4)
        self.assertGreater(len(fd), 0)
        min_d = fracdiff_min_d(prices)
        self.assertTrue(0.0 <= min_d <= 1.0)

    # 11. SampEn, PermEn, Lyapunov
    def test_entropies_and_lyapunov(self):
        series = [math.sin(i * 0.1) for i in range(100)]
        se = sample_entropy(series)
        pe = permutation_entropy(series)
        lyap = lyapunov_exponent(series)
        self.assertGreaterEqual(se, 0.0)
        self.assertTrue(0.0 <= pe <= 1.0)
        self.assertIsInstance(lyap, float)

    # 12. VAR(p) + Granger
    def test_var_fit_and_granger(self):
        rng = random.Random(1)
        x = [rng.gauss(0, 1) for _ in range(100)]
        # y lags x
        y = [0.0] + [0.8 * x[i - 1] + rng.gauss(0, 0.2) for i in range(1, 100)]
        var_res = var_fit({"X": x, "Y": y}, p=1)
        self.assertIn("granger", var_res)
        self.assertIn("X→Y", var_res["granger"])
        self.assertGreater(var_res["granger"]["X→Y"], 1.0)

    # 13. Optimal Exit Threshold (OU)
    def test_optimal_exit_threshold(self):
        ou_p = {"speed": 0.2, "mu": 1.5, "sigma2": 0.04}
        thresh = optimal_exit_threshold(ou_p, risk_aversion=1.0)
        self.assertIn("optimal_exit", thresh)
        self.assertGreater(thresh["optimal_exit"], thresh["ou_mean"])

    # 14. Black-Scholes IV & Greeks
    def test_black_scholes_iv_and_greeks(self):
        # Known option price: S=100, K=100, T=1.0, sigma=0.20 -> Call ≈ 7.9655
        price = black_scholes_price(S=100.0, K=100.0, T=1.0, sigma=0.20)
        self.assertAlmostEqual(price, 7.9655, places=3)
        iv_res = black_scholes_iv(market_price=price, S=100.0, K=100.0, T=1.0)
        self.assertTrue(iv_res["converged"])
        self.assertAlmostEqual(iv_res["iv"], 0.20, places=2)
        self.assertAlmostEqual(iv_res["delta"], 0.5398, places=2)


if __name__ == "__main__":
    unittest.main()
