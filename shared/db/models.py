from sqlalchemy import (
    Column, String, Integer, BigInteger, Numeric, DateTime, Text,
    Index, ForeignKey, UniqueConstraint, Enum as SQLEnum, Boolean, JSON
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, relationship, Mapped, mapped_column
from sqlalchemy.ext.asyncio import AsyncAttrs
from datetime import datetime
from decimal import Decimal
import uuid
from typing import Optional, List
import enum


class Base(AsyncAttrs, DeclarativeBase):
    pass


class OrderStatusEnum(str, enum.Enum):
    DRAFT = "DRAFT"
    RISK_PENDING = "RISK_PENDING"
    RISK_REJECTED = "RISK_REJECTED"
    RISK_APPROVED = "RISK_APPROVED"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    PROTECTION_PENDING = "PROTECTION_PENDING"
    PROTECTED = "PROTECTED"
    EXIT_SUBMITTED = "EXIT_SUBMITTED"
    CLOSED = "CLOSED"
    UNKNOWN = "UNKNOWN"
    RECONCILIATION_REQUIRED = "RECONCILIATION_REQUIRED"
    MANUAL_REVIEW = "MANUAL_REVIEW"


class OrderSideEnum(str, enum.Enum):
    BUY = "buy"
    SELL = "sell"


class OrderTypeEnum(str, enum.Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT_MARKET = "take_profit_market"
    TAKE_PROFIT_LIMIT = "take_profit_limit"


class MarginModeEnum(str, enum.Enum):
    ISOLATED = "isolated"
    CROSSED = "crossed"


class RiskDecisionEnum(str, enum.Enum):
    APPROVED = "approved"
    REJECTED = "rejected"


class KillSwitchLevelEnum(str, enum.Enum):
    YELLOW = "yellow"
    ORANGE = "orange"
    RED = "red"
    BLACK = "black"


class TradeDossier(Base):
    __tablename__ = "trade_dossiers"

    trade_id = Column(String(64), primary_key=True)
    strategy_version = Column(String(64), nullable=False)
    model_version = Column(String(64), nullable=True)
    config_version = Column(String(64), nullable=False)
    market_regime = Column(String(64), nullable=False)
    entry_signal = Column(JSON, nullable=False)
    feature_snapshot = Column(JSON, nullable=False)
    risk_decision = Column(JSON, nullable=False)
    approved_size = Column(Numeric(20, 8), nullable=False)
    entry = Column(JSON, nullable=False)
    exit = Column(JSON, nullable=True)
    sl_tp = Column(JSON, nullable=False)
    order_history = Column(JSON, nullable=False)
    fills = Column(JSON, nullable=False)
    fees_funding_slippage = Column(JSON, nullable=False)
    realized_pnl = Column(Numeric(20, 8), nullable=False)
    exit_reason = Column(String(128), nullable=True)
    loss_classification = Column(String(64), nullable=True)
    technical_incidents = Column(JSON, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    closed_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("ix_trade_dossiers_strategy_created", "strategy_version", "created_at"),
        Index("ix_trade_dossiers_regime_created", "market_regime", "created_at"),
    )


class Order(Base):
    __tablename__ = "orders"

    order_id = Column(String(64), primary_key=True)
    client_order_id = Column(String(64), unique=True, nullable=False, index=True)
    exchange_order_id = Column(String(64), nullable=True, index=True)
    trade_id = Column(String(64), ForeignKey("trade_dossiers.trade_id"), nullable=False)
    pair = Column(String(32), nullable=False, index=True)
    side = Column(SQLEnum(OrderSideEnum), nullable=False)
    order_type = Column(SQLEnum(OrderTypeEnum), nullable=False)
    status = Column(SQLEnum(OrderStatusEnum), nullable=False, default=OrderStatusEnum.DRAFT, index=True)
    amount = Column(Numeric(20, 8), nullable=False)
    filled = Column(Numeric(20, 8), nullable=False, default=0)
    price = Column(Numeric(20, 8), nullable=False)
    avg_price = Column(Numeric(20, 8), nullable=True)
    stop_price = Column(Numeric(20, 8), nullable=True)
    leverage = Column(Integer, nullable=False)
    margin_mode = Column(SQLEnum(MarginModeEnum), nullable=False)
    time_in_force = Column(String(8), nullable=False, default="GTC")
    stop_loss = Column(Numeric(20, 8), nullable=False)
    take_profit = Column(Numeric(20, 8), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    exchange_timestamp = Column(DateTime, nullable=True)
    raw_response = Column(JSON, nullable=False)

    trade_dossier = relationship("TradeDossier", backref="orders")

    __table_args__ = (
        Index("ix_orders_trade_status", "trade_id", "status"),
        Index("ix_orders_pair_created", "pair", "created_at"),
    )


class Fill(Base):
    __tablename__ = "fills"

    fill_id = Column(String(64), primary_key=True)
    order_id = Column(String(64), ForeignKey("orders.order_id"), nullable=False)
    trade_id = Column(String(64), ForeignKey("trade_dossiers.trade_id"), nullable=False)
    pair = Column(String(32), nullable=False)
    side = Column(SQLEnum(OrderSideEnum), nullable=False)
    price = Column(Numeric(20, 8), nullable=False)
    amount = Column(Numeric(20, 8), nullable=False)
    fee = Column(Numeric(20, 8), nullable=False)
    fee_currency = Column(String(16), nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    exchange_timestamp = Column(DateTime, nullable=True)
    trade_type = Column(String(32), nullable=False)
    liquidation = Column(Boolean, nullable=False, default=False)

    __table_args__ = (
        Index("ix_fills_order_timestamp", "order_id", "timestamp"),
        Index("ix_fills_trade_timestamp", "trade_id", "timestamp"),
    )


class Position(Base):
    __tablename__ = "positions"

    position_id = Column(String(64), primary_key=True)
    pair = Column(String(32), nullable=False, index=True)
    side = Column(SQLEnum(OrderSideEnum), nullable=False)
    size = Column(Numeric(20, 8), nullable=False)
    entry_price = Column(Numeric(20, 8), nullable=False)
    mark_price = Column(Numeric(20, 8), nullable=False)
    leverage = Column(Integer, nullable=False)
    margin_mode = Column(SQLEnum(MarginModeEnum), nullable=False)
    unrealized_pnl = Column(Numeric(20, 8), nullable=False, default=0)
    realized_pnl = Column(Numeric(20, 8), nullable=False, default=0)
    liquidation_price = Column(Numeric(20, 8), nullable=True)
    margin_ratio = Column(Numeric(10, 6), nullable=True)
    stop_loss = Column(Numeric(20, 8), nullable=True)
    take_profit = Column(Numeric(20, 8), nullable=True)
    opened_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    exchange_position_id = Column(String(64), nullable=True, unique=True, index=True)

    __table_args__ = (
        Index("ix_positions_pair_updated", "pair", "updated_at"),
    )


class MarketCandle(Base):
    __tablename__ = "market_candles"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    pair = Column(String(32), nullable=False)
    timeframe = Column(String(16), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    open = Column(Numeric(20, 8), nullable=False)
    high = Column(Numeric(20, 8), nullable=False)
    low = Column(Numeric(20, 8), nullable=False)
    close = Column(Numeric(20, 8), nullable=False)
    volume = Column(Numeric(30, 8), nullable=False)
    source = Column(String(32), nullable=False, default="binance")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("pair", "timeframe", "timestamp", name="uq_market_candles_pair_tf_ts"),
        Index("ix_market_candles_pair_tf_ts", "pair", "timeframe", "timestamp"),
        Index("ix_market_candles_ts", "timestamp"),
    )


class MarketSnapshot(Base):
    __tablename__ = "market_snapshots"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    pair = Column(String(32), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    mark_price = Column(Numeric(20, 8), nullable=False)
    index_price = Column(Numeric(20, 8), nullable=False)
    last_price = Column(Numeric(20, 8), nullable=False)
    bid_price = Column(Numeric(20, 8), nullable=False)
    ask_price = Column(Numeric(20, 8), nullable=False)
    bid_size = Column(Numeric(20, 8), nullable=False)
    ask_size = Column(Numeric(20, 8), nullable=False)
    spread = Column(Numeric(20, 8), nullable=False)
    funding_rate = Column(Numeric(20, 8), nullable=True)
    open_interest = Column(Numeric(30, 8), nullable=True)
    volume_24h = Column(Numeric(30, 8), nullable=True)
    source = Column(String(32), nullable=False, default="binance")

    __table_args__ = (
        UniqueConstraint("pair", "timestamp", name="uq_market_snapshots_pair_ts"),
        Index("ix_market_snapshots_pair_ts", "pair", "timestamp"),
    )


class FeatureVector(Base):
    __tablename__ = "features"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    pair = Column(String(32), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    timeframe = Column(String(16), nullable=False)
    feature_version = Column(String(64), nullable=False)
    features = Column(JSON, nullable=False)
    regime = Column(String(64), nullable=True)
    confidence = Column(Numeric(5, 4), nullable=True)

    __table_args__ = (
        Index("ix_features_pair_ts", "pair", "timestamp"),
    )


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    pair = Column(String(32), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    probability = Column(Numeric(5, 4), nullable=False)
    confidence = Column(Numeric(5, 4), nullable=False)
    regime = Column(String(64), nullable=True)
    model_version = Column(String(64), nullable=False)

    __table_args__ = (
        Index("ix_predictions_pair_ts", "pair", "timestamp"),
    )


class Signal(Base):
    __tablename__ = "signals"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    pair = Column(String(32), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    strategy_version = Column(String(64), nullable=False)
    signal_type = Column(String(32), nullable=False)
    reason = Column(Text, nullable=False)
    metadata_json = Column("metadata", JSON, nullable=False)

    __table_args__ = (
        Index("ix_signals_pair_ts", "pair", "timestamp"),
    )


class TradeIntent(Base):
    __tablename__ = "trade_intents"

    trade_id = Column(String(64), primary_key=True)
    client_order_id = Column(String(64), nullable=False)
    strategy_version = Column(String(64), nullable=False)
    model_version = Column(String(64), nullable=True)
    config_version = Column(String(64), nullable=False)
    pair = Column(String(32), nullable=False)
    side = Column(SQLEnum(OrderSideEnum), nullable=False)
    order_type = Column(SQLEnum(OrderTypeEnum), nullable=False)
    amount = Column(Numeric(20, 8), nullable=False)
    price = Column(Numeric(20, 8), nullable=True)
    stop_price = Column(Numeric(20, 8), nullable=True)
    leverage = Column(Integer, nullable=False)
    margin_mode = Column(SQLEnum(MarginModeEnum), nullable=False)
    stop_loss = Column(Numeric(20, 8), nullable=False)
    take_profit = Column(Numeric(20, 8), nullable=True)
    timeframe = Column(String(16), nullable=False)
    regime = Column(String(64), nullable=True)
    signal_metadata = Column(JSON, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class RiskDecision(Base):
    __tablename__ = "risk_decisions"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    trade_id = Column(String(64), ForeignKey("trade_intents.trade_id"), nullable=False)
    decision = Column(SQLEnum(RiskDecisionEnum), nullable=False)
    reason = Column(Text, nullable=False)
    checks = Column(JSON, nullable=False)
    approved_size = Column(Numeric(20, 8), nullable=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)


class Proposal(Base):
    __tablename__ = "proposals"

    proposal_id = Column(String(64), primary_key=True)
    strategy_version = Column(String(64), nullable=False)
    problem_type = Column(String(64), nullable=False)
    evidence = Column(JSON, nullable=False)
    proposed_change = Column(JSON, nullable=False)
    expected_effect = Column(Text, nullable=False)
    validation_plan = Column(Text, nullable=False)
    rollback_condition = Column(Text, nullable=False)
    status = Column(String(32), nullable=False, default="pending")
    experiment_id = Column(String(64), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class Experiment(Base):
    __tablename__ = "experiments"

    experiment_id = Column(String(64), primary_key=True)
    proposal_id = Column(String(64), ForeignKey("proposals.proposal_id"), nullable=False)
    candidate_config = Column(JSON, nullable=False)
    baseline_config = Column(JSON, nullable=False)
    status = Column(String(32), nullable=False, default="pending")
    metrics = Column(JSON, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)


class Deployment(Base):
    __tablename__ = "deployments"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    deployment_id = Column(String(64), unique=True, nullable=False)
    strategy_version = Column(String(64), nullable=False)
    model_version = Column(String(64), nullable=True)
    config_version = Column(String(64), nullable=False)
    environment = Column(String(32), nullable=False)
    status = Column(String(32), nullable=False, default="active")
    deployed_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    rolled_back_at = Column(DateTime, nullable=True)


class AuditEvent(Base):
    __tablename__ = "audit_events"

    event_id = Column(String(64), primary_key=True)
    event_type = Column(String(64), nullable=False)
    actor = Column(String(128), nullable=False)
    resource_type = Column(String(64), nullable=False)
    resource_id = Column(String(64), nullable=False)
    old_value = Column(JSON, nullable=True)
    new_value = Column(JSON, nullable=True)
    metadata_json = Column("metadata", JSON, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_audit_events_ts", "timestamp"),
        Index("ix_audit_events_actor_ts", "actor", "timestamp"),
    )


class Incident(Base):
    __tablename__ = "incidents"

    incident_id = Column(String(64), primary_key=True)
    incident_type = Column(String(64), nullable=False)
    severity = Column(String(16), nullable=False)
    title = Column(String(256), nullable=False)
    description = Column(Text, nullable=True)
    related_ids = Column(JSON, nullable=False)
    status = Column(String(32), nullable=False, default="open")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)
    resolved_by = Column(String(128), nullable=True)


class KillSwitchLog(Base):
    __tablename__ = "kill_switch_log"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    level = Column(SQLEnum(KillSwitchLevelEnum), nullable=False)
    reason = Column(Text, nullable=False)
    activated_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    activated_by = Column(String(128), nullable=False)
    auto_recover = Column(Boolean, nullable=False, default=False)
    recovered_at = Column(DateTime, nullable=True)


# =============================================================================
# Feedback loop pipeline (Fase 1: trade_outcomes source-of-truth)
# =============================================================================
class TradeOutcome(Base):
    __tablename__ = "trade_outcomes"

    trade_id = Column(BigInteger, primary_key=True)
    pair = Column(String(32), nullable=False)
    timeframe = Column(String(8), nullable=False)
    # Snapshot fitur ML + regime + predicted_rr saat entry
    entry_conditions = Column(JSON, nullable=False)
    exit_reason = Column(String(64), nullable=False)
    pnl_pct = Column(Numeric(20, 8), nullable=False)
    pnl_abs = Column(Numeric(20, 8), nullable=False)   # net setelah fee
    predicted_rr = Column(Numeric(12, 6), nullable=True)
    actual_rr = Column(Numeric(12, 6), nullable=True)
    regime_at_entry = Column(String(32), nullable=True)
    timestamp_entry = Column(DateTime, nullable=False)
    timestamp_exit = Column(DateTime, nullable=False)
    # Reconciliation flag — antisipasi event Redis yang hilang (at-most-once)
    processed_by_attribution = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_trade_outcomes_unprocessed", "processed_by_attribution"),
        Index("ix_trade_outcomes_pair_exit", "pair", "timestamp_exit"),
    )


class AttributionResult(Base):
    __tablename__ = "attribution_results"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    trade_id = Column(BigInteger, ForeignKey("trade_outcomes.trade_id"), nullable=False, index=True)
    signal_correct = Column(Boolean, nullable=False)
    drift_contribution = Column(Numeric(12, 6), nullable=True)  # |actual_rr - predicted_rr|
    regime = Column(String(32), nullable=True, index=True)
    feature_importance_snapshot = Column(JSON, nullable=True)
    # ML juga punya label "is_ml_callable" — apakah ML service responsif saat entry
    ml_recommendation = Column(String(16), nullable=True)   # BUY/SELL/HOLD dari ML
    pnl_pct = Column(Numeric(20, 8), nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_attribution_regime_created", "regime", "created_at"),
    )


class ModelVersion(Base):
    __tablename__ = "model_versions"

    version_id = Column(String(64), primary_key=True)
    trained_at = Column(DateTime, nullable=False)
    dataset_hash = Column(String(64), nullable=True)
    holdout_metrics = Column(JSON, nullable=True)
    status = Column(String(16), nullable=False, default="candidate")  # candidate/production/rejected/archived
    promoted_at = Column(DateTime, nullable=True)
    rejected_reason = Column(Text, nullable=True)


class RetrainJob(Base):
    __tablename__ = "retrain_jobs"

    job_id = Column(String(64), primary_key=True)
    triggered_at = Column(DateTime, nullable=False)
    trigger_reason = Column(String(32), nullable=True)
    dataset_size = Column(BigInteger, nullable=True)
    status = Column(String(16), nullable=False, default="running")  # running/completed/failed
    completed_at = Column(DateTime, nullable=True)
    resulting_model_version_id = Column(String(64), ForeignKey("model_versions.version_id"), nullable=True)


class ModelShadowEvaluation(Base):
    __tablename__ = "model_shadow_evaluations"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    version_id = Column(String(64), ForeignKey("model_versions.version_id"), nullable=False, unique=True)
    started_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    ends_at = Column(DateTime, nullable=False)
    candidate_score = Column(Numeric(12, 8), nullable=True)
    champion_score = Column(Numeric(12, 8), nullable=True)
    samples = Column(BigInteger, nullable=False, default=0)
    status = Column(String(16), nullable=False, default="running")
    details = Column(JSON, nullable=False)


class PositionHealthSnapshot(Base):
    __tablename__ = "position_health_snapshots"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    trade_id = Column(String(64), nullable=False, index=True)
    pair = Column(String(32), nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    health_score = Column(Numeric(8, 4), nullable=False)
    thesis_valid = Column(Boolean, nullable=False)
    momentum_decay = Column(Numeric(8, 6), nullable=False)
    regime = Column(String(64), nullable=True)
    details = Column(JSON, nullable=False)

    __table_args__ = (Index("ix_position_health_pair_ts", "pair", "timestamp"),)


class ExitRegret(Base):
    __tablename__ = "exit_regrets"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    trade_id = Column(String(64), nullable=False, unique=True, index=True)
    pair = Column(String(32), nullable=False)
    exit_price = Column(Numeric(20, 8), nullable=False)
    best_future_price = Column(Numeric(20, 8), nullable=True)
    regret_pct = Column(Numeric(12, 8), nullable=False)
    classification = Column(String(32), nullable=False)
    horizon_candles = Column(Integer, nullable=False, default=20)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
