import os
import sys
from decimal import Decimal

import pytest

_SERVICE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "services", "risk-gateway",
)
if _SERVICE_DIR not in sys.path:
    sys.path.insert(0, _SERVICE_DIR)

from engine import RiskGateway
from policy import PolicyConfig


class TestRiskEngine:
    @pytest.mark.asyncio
    async def test_check_leverage(self):
        policy = PolicyConfig(max_leverage=3)
        gateway = RiskGateway(policy)
        res_pass = await gateway.check_leverage(3)
        assert res_pass.passed is True
        res_fail = await gateway.check_leverage(5)
        assert res_fail.passed is False

    @pytest.mark.asyncio
    async def test_check_environment(self):
        policy = PolicyConfig(trading_mode="demo")
        gateway = RiskGateway(policy)
        res_pass = await gateway.check_environment("demo")
        assert res_pass.passed is True
        res_fail = await gateway.check_environment("live")
        assert res_fail.passed is False

    @pytest.mark.asyncio
    async def test_check_exposure_limits(self):
        policy = PolicyConfig(max_total_exposure_pct=0.10)
        gateway = RiskGateway(policy)
        res_pass = await gateway.check_exposure(
            current_exposure=Decimal("500"),
            new_notional=Decimal("400"),
            equity=Decimal("10000"),
        )
        assert res_pass.passed is True
        res_fail = await gateway.check_exposure(
            current_exposure=Decimal("800"),
            new_notional=Decimal("500"),
            equity=Decimal("10000"),
        )
        assert res_fail.passed is False
