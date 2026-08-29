import unittest

from shared.quant import fractional_kelly, regime_threshold, weighted_factor_score
from shared.quant.execution import slippage_bps
from shared.quant.orderbook import dom_pressure
from shared.quant.position_risk import kill_switch_level
from shared.quant.allocation import exposure_multiplier
from shared.quant.position import exit_consensus
from shared.quant.position import position_health
from shared.quant.advanced import autocorrelation, cusum_break, optimal_tp, parkinson_volatility


class QuantContractsTest(unittest.TestCase):
    def test_kelly_is_bounded(self):
        self.assertGreaterEqual(fractional_kelly(.6, 2, 1), 0)
        self.assertLessEqual(fractional_kelly(1, 10, 1), .02)

    def test_regime_threshold(self):
        self.assertGreater(regime_threshold("sideways_high_vol"), regime_threshold("trending_up"))
        self.assertEqual(regime_threshold("TRENDING_BEAR"), regime_threshold("trending_down"))
        self.assertEqual(exposure_multiplier(0, "TRENDING_BULL"), 1.0)

    def test_factor_score_is_bounded(self):
        self.assertEqual(weighted_factor_score({"momentum": 1.0}), 100.0)
        self.assertEqual(weighted_factor_score({"momentum": -1.0}), 0.0)

    def test_execution_and_orderbook(self):
        self.assertAlmostEqual(slippage_bps(100, 101), 100)
        self.assertGreater(dom_pressure(10, 1), 0)

    def test_kill_switch_escalates(self):
        self.assertEqual(kill_switch_level(.10), "black")

    def test_advanced_estimators_are_bounded_or_positive(self):
        self.assertGreater(parkinson_volatility([110, 112], [100, 101]), 0)
        self.assertGreaterEqual(autocorrelation([1, 2, 3, 4, 5]), -1)
        self.assertLessEqual(autocorrelation([1, 2, 3, 4, 5]), 1)
        self.assertIn("break", cusum_break([0.01] * 12))
        self.assertEqual(optimal_tp([]), 1.5)

    def test_exit_consensus_requires_multiple_confirmations(self):
        should_exit, score = exit_consensus({"regime": True, "momentum": True, "reversal": True, "volume": True})
        self.assertTrue(should_exit)
        self.assertGreaterEqual(score, .65)
        should_exit, _ = exit_consensus({"regime": True})
        self.assertFalse(should_exit)

    def test_position_health_tracks_reversion_from_peak(self):
        healthy = position_health(2.0, 2.0, "TRENDING_BULL", "TRENDING_BULL", 1.0)
        decayed = position_health(0.5, 2.0, "TRENDING_BEAR", "TRENDING_BULL", 0.8)
        self.assertEqual(healthy["momentum_decay"], 0.0)
        self.assertGreater(decayed["momentum_decay"], 0.0)
        self.assertFalse(decayed["thesis_valid"])


if __name__ == "__main__":
    unittest.main()
