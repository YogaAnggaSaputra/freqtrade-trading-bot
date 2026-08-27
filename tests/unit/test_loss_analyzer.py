import os
import sys
from decimal import Decimal

import pytest

_SERVICE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "services", "loss-analyzer",
)
if _SERVICE_DIR not in sys.path:
    sys.path.insert(0, _SERVICE_DIR)

from analyzer import LossAnalyzer

from shared.db.models import TradeDossier


class TestLossAnalyzer:
    @pytest.mark.asyncio
    async def test_classify_trade_loss_winner(self):
        analyzer = LossAnalyzer()
        dossier = TradeDossier(
            realized_pnl=Decimal("10.0"),
        )
        res = await analyzer.classify_trade_loss(dossier)
        assert res == "winner"

    @pytest.mark.asyncio
    async def test_classify_trade_loss_technical_failure(self):
        analyzer = LossAnalyzer()
        dossier = TradeDossier(
            realized_pnl=Decimal("-10.0"),
            technical_incidents=[{"type": "api_timeout"}],
        )
        res = await analyzer.classify_trade_loss(dossier)
        assert res == "technical_failure"

    @pytest.mark.asyncio
    async def test_classify_trade_loss_sl_noise(self):
        analyzer = LossAnalyzer()
        dossier = TradeDossier(
            realized_pnl=Decimal("-10.0"),
            exit_reason="stop_loss",
            entry={"price": 100},
            sl_tp={"stop_loss": 99.5},
            technical_incidents=[],
        )
        res = await analyzer.classify_trade_loss(dossier)
        assert res == "sl_noise"
