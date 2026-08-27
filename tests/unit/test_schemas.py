from datetime import UTC, datetime
from decimal import Decimal

from shared.schemas import (
    AuditEvent,
    Fill,
    KillSwitchLevel,
    KillSwitchState,
    MarginMode,
    MarketCandle,
    MarketSnapshot,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    PositionSide,
    Proposal,
    RiskCheckResult,
    RiskDecision,
    TimeInForce,
    TradeDossier,
    TradeIntent,
)


class TestSchemas:
    def test_order_creation(self):
        order = Order(
            order_id="ord_123",
            client_order_id="clord_123",
            trade_id="trade_123",
            pair="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            status=OrderStatus.DRAFT,
            amount=Decimal("0.1"),
            filled=Decimal("0"),
            price=Decimal("50000"),
            leverage=3,
            margin_mode=MarginMode.ISOLATED,
            time_in_force=TimeInForce.GTC,
            stop_loss=Decimal("49000"),
            take_profit=Decimal("52000"),
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        assert order.order_id == "ord_123"
        assert order.side == OrderSide.BUY
        assert order.amount == Decimal("0.1")

    def test_position_creation(self):
        position = Position(
            position_id="pos_123",
            pair="BTCUSDT",
            side=PositionSide.LONG,
            size=Decimal("0.1"),
            entry_price=Decimal("50000"),
            mark_price=Decimal("50500"),
            leverage=3,
            margin_mode=MarginMode.ISOLATED,
            unrealized_pnl=Decimal("50"),
            realized_pnl=Decimal("0"),
            opened_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        assert position.side == PositionSide.LONG
        assert position.size == Decimal("0.1")

    def test_fill_creation(self):
        fill = Fill(
            fill_id="fill_123",
            order_id="ord_123",
            trade_id="trade_123",
            pair="BTCUSDT",
            side=OrderSide.BUY,
            price=Decimal("50000"),
            amount=Decimal("0.1"),
            fee=Decimal("5"),
            fee_currency="USDT",
            timestamp=datetime.now(UTC),
            trade_type="entry",
            liquidation=False,
        )
        assert fill.amount == Decimal("0.1")
        assert fill.fee == Decimal("5")

    def test_market_candle_creation(self):
        candle = MarketCandle(
            pair="BTCUSDT",
            timeframe="15m",
            timestamp=datetime.now(UTC),
            open=Decimal("49900"),
            high=Decimal("50100"),
            low=Decimal("49800"),
            close=Decimal("50000"),
            volume=Decimal("100.5"),
        )
        assert candle.high == Decimal("50100")
        assert candle.low == Decimal("49800")

    def test_market_snapshot_creation(self):
        snapshot = MarketSnapshot(
            pair="BTCUSDT",
            timestamp=datetime.now(UTC),
            mark_price=Decimal("50000"),
            index_price=Decimal("50001"),
            last_price=Decimal("50000"),
            bid_price=Decimal("49999"),
            ask_price=Decimal("50001"),
            bid_size=Decimal("10"),
            ask_size=Decimal("15"),
            spread=Decimal("2"),
            funding_rate=Decimal("0.0001"),
            open_interest=Decimal("1000000"),
            volume_24h=Decimal("50000000"),
        )
        assert snapshot.spread == Decimal("2")

    def test_trade_intent_creation(self):
        intent = TradeIntent(
            trade_id="trade_123",
            client_order_id="clord_123",
            strategy_version="EmaTrendV1",
            config_version="v1.0.0",
            pair="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=Decimal("0.1"),
            price=Decimal("50000"),
            leverage=3,
            margin_mode=MarginMode.ISOLATED,
            stop_loss=Decimal("49000"),
            take_profit=Decimal("52000"),
            timeframe="15m",
            regime="trending_up",
            signal_metadata={"ema_cross": True, "adx": 25},
        )
        assert intent.strategy_version == "EmaTrendV1"
        assert intent.regime == "trending_up"

    def test_risk_check_result(self):
        result = RiskCheckResult(
            decision=RiskDecision.APPROVED,
            reason="All checks passed",
            checks=[
                {"name": "kill_switch", "passed": True, "reason": "No active kill switch"},
                {"name": "pair_allowlist", "passed": True, "reason": "BTCUSDT allowed"},
            ],
            approved_size=Decimal("0.1"),
        )
        assert result.decision == RiskDecision.APPROVED
        assert result.approved_size == Decimal("0.1")

    def test_proposal_creation(self):
        proposal = Proposal(
            proposal_id="HERMES-2026-07-30-001",
            strategy_version="EmaTrendV1",
            problem_type="regime_mismatch",
            evidence={"sample_size": 50, "net_pnl": -0.02, "max_drawdown": 0.04},
            proposed_change={
                "class": "safe_experiment",
                "parameter": "adx_min",
                "old_value": 15,
                "new_value": 20,
            },
            expected_effect="reduce low-quality trend entries",
            validation_plan="walk_forward_6_windows_then_demo_14d",
            rollback_condition="canary_drawdown_gt_baseline_by_1_percent",
        )
        assert proposal.proposal_id == "HERMES-2026-07-30-001"
        assert proposal.status == "pending"

    def test_kill_switch_state(self):
        ks = KillSwitchState(
            level=KillSwitchLevel.RED,
            reason="SL not active on position",
            activated_at=datetime.now(UTC),
            activated_by="risk_gateway",
            auto_recover=False,
        )
        assert ks.level == KillSwitchLevel.RED
        assert ks.auto_recover is False

    def test_trade_dossier_creation(self):
        dossier = TradeDossier(
            trade_id="trade_123",
            strategy_version="EmaTrendV1",
            config_version="v1.0.0",
            market_regime="trending_up",
            entry_signal={"ema_cross_up": True, "adx": 25},
            feature_snapshot={"ema_fast": 50000, "ema_slow": 49950},
            risk_decision={"decision": "approved", "reason": "All checks passed"},
            approved_size=Decimal("0.1"),
            entry={"price": 50000, "size": 0.1, "timestamp": "2026-07-30T12:00:00Z"},
            exit={"price": 50500, "size": 0.1, "timestamp": "2026-07-30T14:00:00Z"},
            sl_tp={"stop_loss": 49000, "take_profit": 52000},
            order_history=[],
            fills=[],
            fees_funding_slippage={"fees": 5, "funding": 0.1, "slippage": 0.5},
            realized_pnl=Decimal("45"),
            exit_reason="take_profit",
            loss_classification=None,
            technical_incidents=[],
            created_at=datetime.now(UTC),
            closed_at=datetime.now(UTC),
        )
        assert dossier.realized_pnl == Decimal("45")
        assert dossier.exit_reason == "take_profit"

    def test_audit_event_creation(self):
        event = AuditEvent(
            event_id="audit_123",
            event_type="strategy_deployed",
            actor="admin:john",
            resource_type="deployment",
            resource_id="deploy_123",
            old_value={"strategy_version": "EmaTrendV0"},
            new_value={"strategy_version": "EmaTrendV1"},
            metadata={"environment": "demo"},
        )
        assert event.event_type == "strategy_deployed"
        assert event.actor == "admin:john"


class TestEnumValues:
    def test_order_status_enum(self):
        assert OrderStatus.DRAFT.value == "DRAFT"
        assert OrderStatus.RISK_APPROVED.value == "RISK_APPROVED"
        assert OrderStatus.PROTECTED.value == "PROTECTED"
        assert OrderStatus.RECONCILIATION_REQUIRED.value == "RECONCILIATION_REQUIRED"

    def test_kill_switch_level_enum(self):
        assert KillSwitchLevel.YELLOW.value == "yellow"
        assert KillSwitchLevel.ORANGE.value == "orange"
        assert KillSwitchLevel.RED.value == "red"
        assert KillSwitchLevel.BLACK.value == "black"

    def test_margin_mode_enum(self):
        assert MarginMode.ISOLATED.value == "isolated"
        assert MarginMode.CROSSED.value == "crossed"
