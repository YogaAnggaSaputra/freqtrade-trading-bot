import os
import sys
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

# Services are run as plain modules (python main.py inside each folder), so
# the test needs the folder on sys.path instead of a dotted package import.
_SERVICE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "services", "risk-gateway",
)
if _SERVICE_DIR not in sys.path:
    sys.path.insert(0, _SERVICE_DIR)

from engine import RiskGateway
from policy import PolicyConfig


class TestRiskGateway:
    @pytest_asyncio.fixture
    async def risk_gateway(self):
        policy = PolicyConfig(
            trading_mode="demo",
            pair_allowlist=["BTCUSDT", "ETHUSDT"],
            max_leverage=3,
            risk_per_trade_pct=0.005,
            max_total_exposure_pct=0.10,
            max_open_trades=1,
            daily_loss_limit_pct=0.02,
            max_drawdown_pct=0.05,
            stoploss_mandatory=True,
        )
        gateway = RiskGateway(policy)
        gateway._redis = AsyncMock()
        yield gateway

    @pytest.mark.asyncio
    async def test_check_kill_switch_inactive(self, risk_gateway):
        risk_gateway._redis.get = AsyncMock(return_value=None)
        result = await risk_gateway.check_kill_switch()
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_check_kill_switch_active(self, risk_gateway):
        risk_gateway._redis.get = AsyncMock(return_value="red")
        result = await risk_gateway.check_kill_switch()
        assert result.passed is False
        assert "red" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_check_pair_allowlist_valid(self, risk_gateway):
        result = await risk_gateway.check_pair_allowlist("BTCUSDT")
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_check_pair_allowlist_invalid(self, risk_gateway):
        result = await risk_gateway.check_pair_allowlist("SOLUSDT")
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_check_stop_loss_valid_long(self, risk_gateway):
        result = await risk_gateway.check_stop_loss_valid(
            Decimal("49000"), Decimal("50000"), "buy"
        )
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_check_stop_loss_invalid_long(self, risk_gateway):
        result = await risk_gateway.check_stop_loss_valid(
            Decimal("51000"), Decimal("50000"), "buy"
        )
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_check_stop_loss_valid_short(self, risk_gateway):
        result = await risk_gateway.check_stop_loss_valid(
            Decimal("51000"), Decimal("50000"), "sell"
        )
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_check_risk_per_trade_within_limit(self, risk_gateway):
        result = await risk_gateway.check_risk_per_trade(
            Decimal("10000"), Decimal("40")
        )
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_check_risk_per_trade_exceeds_limit(self, risk_gateway):
        result = await risk_gateway.check_risk_per_trade(
            Decimal("10000"), Decimal("100")
        )
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_check_leverage_valid(self, risk_gateway):
        result = await risk_gateway.check_leverage(3)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_check_leverage_invalid(self, risk_gateway):
        result = await risk_gateway.check_leverage(5)
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_check_exposure_within_limit(self, risk_gateway):
        result = await risk_gateway.check_exposure(
            Decimal("500"), Decimal("400"), Decimal("10000")
        )
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_check_exposure_exceeds_limit(self, risk_gateway):
        result = await risk_gateway.check_exposure(
            Decimal("800"), Decimal("500"), Decimal("10000")
        )
        assert result.passed is False


class TestPolicyConfig:
    def test_default_policy(self):
        policy = PolicyConfig()
        assert policy.trading_mode == "demo"
        assert policy.max_leverage == 3
        assert policy.pair_allowlist == ["BTCUSDT"]

    def test_custom_policy(self):
        policy = PolicyConfig(
            trading_mode="live",
            max_leverage=5,
            pair_allowlist=["BTCUSDT", "ETHUSDT"],
        )
        assert policy.trading_mode == "live"
        assert policy.max_leverage == 5
        assert "ETHUSDT" in policy.pair_allowlist


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
