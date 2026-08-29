"""Tests for microstructure math and advanced statistical metrics."""

import math
import random
import unittest

from shared.quant.microstructure import (
    bulk_volume_classification, vpin_toxicity, kyle_lambda, order_flow_imbalance,
)
from shared.quant.stats_advanced import (
    mahalanobis_distance_2d, evt_pareto_tail_index, kolmogorov_smirnov_2sample,
    wasserstein_distance_1d, implementation_shortfall,
)


class MicrostructureTests(unittest.TestCase):
    def test_bulk_volume_classification(self):
        v_buy, v_sell = bulk_volume_classification(10.0, 2.0, 100.0)
        self.assertAlmostEqual(v_buy + v_sell, 100.0, places=9)
        self.assertGreater(v_buy, v_sell)

    def test_vpin_toxicity(self):
        balanced = [(100.0, 100.0)] * 5
        toxic = [(190.0, 10.0)] * 5
        self.assertAlmostEqual(vpin_toxicity(balanced), 0.0, places=9)
        self.assertAlmostEqual(vpin_toxicity(toxic), 0.9, places=9)

    def test_kyle_lambda_recovers_slope(self):
        prices = [0.1, 0.2, 0.3, 0.4, 0.5]
        volumes = [10.0, 20.0, 30.0, 40.0, 50.0]
        lambda_val = kyle_lambda(prices, volumes)
        self.assertAlmostEqual(lambda_val, 0.01, places=9)

    def test_order_flow_imbalance(self):
        # Bid price increases -> positive bid flow
        ofi = order_flow_imbalance(
            bids_top=[(101.0, 5.0)], asks_top=[(102.0, 5.0)],
            prev_bids_top=[(100.0, 5.0)], prev_asks_top=[(102.0, 5.0)]
        )
        self.assertEqual(ofi, 5.0)


class AdvancedStatsTests(unittest.TestCase):
    def test_mahalanobis_distance(self):
        rng = random.Random(42)
        sample = [(rng.gauss(0, 1), rng.gauss(0, 1)) for _ in range(200)]
        normal_point = (0.1, 0.1)
        outlier_point = (5.0, 5.0)
        d_normal = mahalanobis_distance_2d(normal_point, sample)
        d_outlier = mahalanobis_distance_2d(outlier_point, sample)
        self.assertLess(d_normal, 2.0)
        self.assertGreater(d_outlier, 4.0)

    def test_evt_pareto_tail_index(self):
        # Heavy-tailed Pareto returns vs Normal returns
        rng = random.Random(7)
        normal_returns = [rng.gauss(0, 0.01) for _ in range(500)]
        evt_res = evt_pareto_tail_index(normal_returns)
        self.assertIn("xi", evt_res)
        self.assertIn("n_exceedances", evt_res)

    def test_kolmogorov_smirnov(self):
        rng = random.Random(12)
        s1 = [rng.gauss(0, 1) for _ in range(200)]
        s2 = [rng.gauss(0, 1) for _ in range(200)]
        s3 = [rng.gauss(3, 1) for _ in range(200)]
        ks_same = kolmogorov_smirnov_2sample(s1, s2)
        ks_diff = kolmogorov_smirnov_2sample(s1, s3)
        self.assertFalse(ks_same["is_drifted"])
        self.assertTrue(ks_diff["is_drifted"])
        self.assertGreater(ks_diff["d_statistic"], ks_same["d_statistic"])

    def test_wasserstein_distance(self):
        s1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        s2 = [2.0, 3.0, 4.0, 5.0, 6.0]
        w_dist = wasserstein_distance_1d(s1, s2)
        self.assertAlmostEqual(w_dist, 1.0, places=9)

    def test_implementation_shortfall(self):
        isf = implementation_shortfall(
            decision_price=100.0, arrival_price=100.10, execution_price=100.20, side="buy", fee_bps=4.0
        )
        self.assertEqual(isf["delay_cost_bps"], 10.0)
        self.assertAlmostEqual(isf["slippage_bps"], 9.99, places=2)
        self.assertEqual(isf["fee_cost_bps"], 4.0)
        self.assertAlmostEqual(isf["total_shortfall_bps"], 23.99, places=2)
        self.assertEqual(isf["execution_quality"], "acceptable")


if __name__ == "__main__":
    unittest.main()
