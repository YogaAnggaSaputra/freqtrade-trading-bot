"""Unit tests for On-Chain & Off-Chain quantitative modules."""

import unittest
from shared.quant.onchain_offchain import (
    whale_netflow_score, mvrv_zscore, nvt_signal, macro_spillover_index,
    recency_weighted_sentiment, defiliquidation_cascade_risk,
)


class OnChainOffChainMathTests(unittest.TestCase):
    def test_whale_netflow_score(self):
        inflow = whale_netflow_score(inflow_volume=1000.0, outflow_volume=200.0, average_volume=500.0)
        outflow = whale_netflow_score(inflow_volume=200.0, outflow_volume=1000.0, average_volume=500.0)
        self.assertGreater(inflow["netflow_score"], 0.0)
        self.assertEqual(inflow["signal"], "bearish_deposit")
        self.assertLess(outflow["netflow_score"], 0.0)
        self.assertEqual(outflow["signal"], "bullish_withdrawal")

    def test_mvrv_and_nvt(self):
        z = mvrv_zscore(market_cap=2.0e11, realized_cap=1.0e11, historical_mvrv_std=0.5)
        nvt = nvt_signal(market_cap=2.0e11, daily_transacted_volume_90d_sma=5.0e9)
        self.assertAlmostEqual(z, 2.0, places=4)
        self.assertAlmostEqual(nvt, 40.0, places=4)

    def test_macro_spillover_index(self):
        cr = [0.01, -0.02, 0.03, -0.01, 0.02]
        dx = [-0.01, 0.02, -0.03, 0.01, -0.02]
        sp = [0.01, -0.01, 0.02, -0.01, 0.015]
        spillover = macro_spillover_index(cr, dx, sp)
        self.assertIn("dxy_correlation", spillover)
        self.assertIn("sp500_correlation", spillover)
        self.assertLess(spillover["dxy_correlation"], 0.0)

    def test_recency_weighted_sentiment(self):
        docs = [
            {"text": "BTC ETF approval and partnership confirmed!", "timestamp_age_hours": 1.0},
            {"text": "Minor hack reported on small exchange", "timestamp_age_hours": 12.0},
        ]
        res = recency_weighted_sentiment(docs, half_life_hours=6.0, target_symbol="BTC")
        self.assertGreater(res["sentiment_score"], 0.0)
        self.assertEqual(res["label"], "bullish")

    def test_defiliquidation_cascade_risk(self):
        positions = [
            {"liquidation_price": 96.0, "collateral_usdt": 5000000.0},
            {"liquidation_price": 91.0, "collateral_usdt": 15000000.0},
            {"liquidation_price": 75.0, "collateral_usdt": 50000000.0},
        ]
        cascade = defiliquidation_cascade_risk(current_price=100.0, debt_positions=positions)
        self.assertEqual(cascade["at_risk_5pct_usdt"], 5000000.0)
        self.assertEqual(cascade["at_risk_10pct_usdt"], 20000000.0)
        self.assertEqual(cascade["cascade_threat"], "high")


if __name__ == "__main__":
    unittest.main()
