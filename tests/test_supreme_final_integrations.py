"""Unit tests for Supreme Final integrations across all remaining microservices."""

import unittest
from shared.quant.supreme_final import (
    tfidf_decay_sentiment, counterfactual_exit_regret, KalmanReconciler,
    ThompsonProposalSelector, pareto_front, chandelier_exit_ratchet,
)


class SupremeFinalIntegrationsTests(unittest.TestCase):
    def test_tfidf_decay_sentiment(self):
        score_fresh = tfidf_decay_sentiment("BTC ETF approval confirmed", {"approval": 0.5, "etf": 0.6}, {}, elapsed_minutes=0)
        score_old = tfidf_decay_sentiment("BTC ETF approval confirmed", {"approval": 0.5, "etf": 0.6}, {}, elapsed_minutes=120, half_life_min=60)
        self.assertGreater(score_fresh, 0.0)
        self.assertLess(score_old, score_fresh)

    def test_counterfactual_exit_regret(self):
        res = counterfactual_exit_regret([101.0, 103.0, 102.0, 105.0], exit_price=100.0, side="long")
        self.assertEqual(res["mfe"], 0.05)
        self.assertEqual(res["mae"], 0.01)
        self.assertGreater(res["decayed_regret"], 0.0)

    def test_kalman_reconciler(self):
        rec = KalmanReconciler()
        res1 = rec.update(0.0001)
        self.assertFalse(res1["is_anomaly"])
        res2 = rec.update(500.0)
        self.assertTrue(res2["is_anomaly"])

    def test_thompson_proposal_selector(self):
        mab = ThompsonProposalSelector(["param_tune", "risk_reduce"])
        mab.record_outcome("param_tune", True)
        mab.record_outcome("param_tune", True)
        mab.record_outcome("risk_reduce", False)
        picked = mab.select(seed=42)
        self.assertEqual(picked, "param_tune")

    def test_pareto_front(self):
        trials = [
            {"trial": 1, "sharpe": 1.5, "max_drawdown": 0.10},
            {"trial": 2, "sharpe": 2.0, "max_drawdown": 0.15},
            {"trial": 3, "sharpe": 1.0, "max_drawdown": 0.20},  # Dominated by 1 & 2
        ]
        pf = pareto_front(trials)
        self.assertEqual(len(pf), 2)
        self.assertNotIn(3, [t["trial"] for t in pf])

    def test_chandelier_exit_ratchet(self):
        # Long stoploss must only ratchet UP
        stop1 = chandelier_exit_ratchet(highest_price=100.0, current_atr=2.0, previous_stop=90.0, side="long")
        self.assertEqual(stop1, 94.0)
        stop2 = chandelier_exit_ratchet(highest_price=98.0, current_atr=2.0, previous_stop=94.0, side="long")
        self.assertEqual(stop2, 94.0)


if __name__ == "__main__":
    unittest.main()
