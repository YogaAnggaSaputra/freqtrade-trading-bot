import os
import sys
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_RISK_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "services", "risk-gateway",
)
_HERMES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "services", "hermes-agent",
)
_LOSS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "services", "loss-analyzer",
)

for d in (_RISK_DIR, _HERMES_DIR, _LOSS_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

from analyzer import LossAnalyzer
from engine import RiskGateway
from policy import PolicyConfig
from proposal_generator import ProposalGenerator

from shared.schemas import MarginMode, OrderSide, OrderType, RiskDecision, TradeIntent


class TestFullTradeFlowE2E:
    @pytest.mark.asyncio
    async def test_full_autonomous_loop(self):
        # 1. Setup Risk Policy Gateway
        policy = PolicyConfig(
            trading_mode="demo",
            pair_allowlist=["BTCUSDT"],
            max_leverage=3,
            risk_per_trade_pct=0.01,
            max_total_exposure_pct=0.10,
            stoploss_mandatory=True,
        )
        risk_gateway = RiskGateway(policy)
        risk_gateway._redis = AsyncMock()

        # Configure redis to return a fresh timestamp for market data freshness check
        from datetime import datetime
        def mock_redis_get(key):
            if key == "market:last_update:BTCUSDT":
                return datetime.utcnow().isoformat()
            return None
        risk_gateway._redis.get = AsyncMock(side_effect=mock_redis_get)

        # 2. Simulate TradeIntent (from Freqtrade Strategy)
        intent = TradeIntent(
            trade_id="trade_e2e_001",
            client_order_id="clord_e2e_001",
            strategy_version="AITradingStrategy",
            config_version="v1.0.0",
            pair="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=Decimal("0.01"),
            price=Decimal("50000"),
            leverage=3,
            margin_mode=MarginMode.ISOLATED,
            stop_loss=Decimal("49500"),
            take_profit=Decimal("51000"),
        )

        # Mock database select checks (no unreconciled orders, active deployments, consistency)
        with patch("engine.AsyncSessionLocal") as mock_db:
            mock_session = AsyncMock()
            mock_db.return_value.__aenter__.return_value = mock_session

            # Setup queries return values using normal MagicMock for synchronous method chains
            mock_result = MagicMock()
            mock_result.scalars.return_value.all.return_value = []
            mock_result.scalar_one_or_none.return_value = MagicMock()
            mock_result.scalar.return_value = Decimal("0")

            mock_session.execute = AsyncMock(return_value=mock_result)

            # Validate trade intent
            result = await risk_gateway.validate_trade(
                trade_id=intent.trade_id,
                client_order_id=intent.client_order_id,
                strategy_version=intent.strategy_version,
                config_version=intent.config_version,
                pair=intent.pair,
                side=intent.side.value,
                order_type=intent.order_type.value,
                amount=intent.amount,
                price=intent.price,
                leverage=intent.leverage,
                margin_mode=intent.margin_mode.value,
                stop_loss=intent.stop_loss,
                take_profit=intent.take_profit,
                timeframe=intent.timeframe,
                equity=Decimal("10000"),
                trade_mode="demo",
            )
            assert result.decision == RiskDecision.APPROVED

        # 3. Simulate trade execution ending in a Stop Loss hit (Loss Analyzer)
        from shared.db.models import TradeDossier
        dossier = TradeDossier(
            trade_id="trade_e2e_001",
            strategy_version="AITradingStrategy",
            config_version="v1.0.0",
            market_regime="sideways_high_volatility",
            entry_signal={"ema_cross": True},
            feature_snapshot={"regime": "sideways_high_volatility"},
            risk_decision={"decision": "approved"},
            approved_size=Decimal("0.01"),
            entry={"price": 50000},
            sl_tp={"stop_loss": 49950}, # very tight stop loss
            realized_pnl=Decimal("-5"),
            exit_reason="stop_loss",
            technical_incidents=[],
        )

        analyzer = LossAnalyzer()
        loss_class = await analyzer.classify_trade_loss(dossier)
        assert loss_class == "sl_noise"

        # 4. Simulate Hermes Agent analyzing loss and producing proposal
        generator = ProposalGenerator(strategy_version="AITradingStrategy")
        proposals = generator.generate_all(
            loss_summary={
                "sample_size": 25,
                "net_pnl": -150.0,
                "win_rate": 0.35,
                "loss_streak": 3,
                "exit_reason_breakdown": {"stop_loss": 18, "exit_signal": 7},
            },
            regime_performance={"sideways_high_volatility": {"sample_size": 12, "net_pnl": -80.0, "win_rate": 0.30}},
            incidents=[],
            calibration={},
            previous_proposals=[],
        )

        assert len(proposals) > 0
        proposal = proposals[0]
        assert proposal.proposed_change.change_class == "safe_experiment"
        assert proposal.proposed_change.parameter == "adx_min_threshold"
        assert proposal.proposed_change.new_value == 20
