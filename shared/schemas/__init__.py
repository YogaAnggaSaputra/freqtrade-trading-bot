"""
shared/schemas/__init__.py
============================
Pydantic v2 schemas shared across all services.

These models mirror the SQLAlchemy models in `shared.db.models` and are used
for API request/response bodies and message-bus payloads.
"""
from __future__ import annotations

import enum
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class OrderSide(str, enum.Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, enum.Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT_MARKET = "take_profit_market"
    TAKE_PROFIT_LIMIT = "take_profit_limit"


class OrderStatus(str, enum.Enum):
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


class TimeInForce(str, enum.Enum):
    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"
    GTX = "GTX"


class MarginMode(str, enum.Enum):
    ISOLATED = "isolated"
    CROSSED = "crossed"


class PositionSide(str, enum.Enum):
    LONG = "long"
    SHORT = "short"


class RiskDecision(str, enum.Enum):
    APPROVED = "approved"
    REJECTED = "rejected"


class KillSwitchLevel(str, enum.Enum):
    YELLOW = "yellow"
    ORANGE = "orange"
    RED = "red"
    BLACK = "black"


# Aliases matching the DB enums (used by telegram-bot / risk-gateway engine).
OrderStatusEnum = OrderStatus
KillSwitchLevelEnum = KillSwitchLevel
RiskDecisionEnum = RiskDecision


# ---------------------------------------------------------------------------
# Base model config
# ---------------------------------------------------------------------------
class Model(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        arbitrary_types_allowed=True,
    )

    def model_dump(self, *args, **kwargs):  # type: ignore[override]
        kwargs.setdefault("mode", "json")
        return super().model_dump(*args, **kwargs)


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------
class Order(Model):
    order_id: str
    client_order_id: str
    exchange_order_id: Optional[str] = None
    trade_id: str = ""
    pair: str
    side: OrderSide
    order_type: OrderType
    status: OrderStatus = OrderStatus.DRAFT
    amount: Decimal = Decimal("0")
    filled: Decimal = Decimal("0")
    price: Decimal = Decimal("0")
    avg_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    leverage: int = 1
    margin_mode: MarginMode = MarginMode.ISOLATED
    time_in_force: TimeInForce = TimeInForce.GTC
    stop_loss: Decimal = Decimal("0")
    take_profit: Optional[Decimal] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    exchange_timestamp: Optional[datetime] = None
    raw_response: Optional[Dict[str, Any]] = None


class Position(Model):
    position_id: str
    pair: str
    side: PositionSide
    size: Decimal = Decimal("0")
    entry_price: Decimal = Decimal("0")
    mark_price: Decimal = Decimal("0")
    leverage: int = 1
    margin_mode: MarginMode = MarginMode.ISOLATED
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    liquidation_price: Optional[Decimal] = None
    margin_ratio: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    opened_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    exchange_position_id: Optional[str] = None


class Fill(Model):
    fill_id: str
    order_id: str
    trade_id: str = ""
    pair: str
    side: OrderSide
    price: Decimal = Decimal("0")
    amount: Decimal = Decimal("0")
    fee: Decimal = Decimal("0")
    fee_currency: str = "USDT"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    exchange_timestamp: Optional[datetime] = None
    trade_type: str = ""
    liquidation: bool = False


class MarketCandle(Model):
    pair: str
    timeframe: str
    timestamp: datetime
    open: Decimal = Decimal("0")
    high: Decimal = Decimal("0")
    low: Decimal = Decimal("0")
    close: Decimal = Decimal("0")
    volume: Decimal = Decimal("0")
    source: str = "binance"


class MarketSnapshot(Model):
    pair: str
    timestamp: datetime
    mark_price: Decimal = Decimal("0")
    index_price: Decimal = Decimal("0")
    last_price: Decimal = Decimal("0")
    bid_price: Decimal = Decimal("0")
    ask_price: Decimal = Decimal("0")
    bid_size: Decimal = Decimal("0")
    ask_size: Decimal = Decimal("0")
    spread: Decimal = Decimal("0")
    funding_rate: Optional[Decimal] = None
    open_interest: Optional[Decimal] = None
    volume_24h: Optional[Decimal] = None
    source: str = "binance"


class TradeIntent(Model):
    trade_id: str
    client_order_id: str
    strategy_version: str
    config_version: str
    pair: str
    side: OrderSide
    order_type: OrderType
    amount: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    leverage: int = 1
    margin_mode: MarginMode = MarginMode.ISOLATED
    stop_loss: Decimal
    take_profit: Optional[Decimal] = None
    timeframe: str = "5m"
    regime: Optional[str] = None
    signal_metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CheckResult(Model):
    name: str
    passed: bool
    reason: str = ""
    details: Optional[Dict[str, Any]] = None


class RiskCheckResult(Model):
    decision: RiskDecision
    reason: str = ""
    checks: List[CheckResult] = Field(default_factory=list)
    approved_size: Optional[Decimal] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HealthCheck(Model):
    service: str
    status: str = "healthy"
    checks: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Proposal(Model):
    proposal_id: str
    strategy_version: str
    problem_type: str
    evidence: Dict[str, Any] = Field(default_factory=dict)
    proposed_change: Dict[str, Any] = Field(default_factory=dict)
    expected_effect: str = ""
    validation_plan: str = ""
    rollback_condition: str = ""
    status: str = "pending"
    experiment_id: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Experiment(Model):
    experiment_id: str
    proposal_id: str
    candidate_config: Dict[str, Any] = Field(default_factory=dict)
    baseline_config: Dict[str, Any] = Field(default_factory=dict)
    status: str = "pending"
    metrics: Dict[str, Any] = Field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class KillSwitchState(Model):
    level: KillSwitchLevel = KillSwitchLevel.YELLOW
    reason: str = ""
    activated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    activated_by: str = ""
    auto_recover: bool = False


class TradeDossier(Model):
    trade_id: str
    strategy_version: str
    config_version: str
    market_regime: str
    entry_signal: Dict[str, Any] = Field(default_factory=dict)
    feature_snapshot: Dict[str, Any] = Field(default_factory=dict)
    risk_decision: Dict[str, Any] = Field(default_factory=dict)
    approved_size: Decimal = Decimal("0")
    entry: Dict[str, Any] = Field(default_factory=dict)
    exit: Optional[Dict[str, Any]] = None
    sl_tp: Dict[str, Any] = Field(default_factory=dict)
    order_history: List[Any] = Field(default_factory=list)
    fills: List[Any] = Field(default_factory=list)
    fees_funding_slippage: Dict[str, Any] = Field(default_factory=dict)
    realized_pnl: Decimal = Decimal("0")
    exit_reason: Optional[str] = None
    loss_classification: Optional[str] = None
    technical_incidents: List[Any] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    closed_at: Optional[datetime] = None


class AuditEvent(Model):
    event_id: str
    event_type: str
    actor: str
    resource_type: str
    resource_id: str
    old_value: Optional[Dict[str, Any]] = None
    new_value: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class OBISnapshot(Model):
    """Order Book Imbalance snapshot dari orderbook depth real-time."""
    pair: str
    obi: float = 0.0                          # [-1, 1]: positif = tekanan beli dominan
    bid_volume: float = 0.0
    ask_volume: float = 0.0
    liquidity_sweep: str = "none"             # "none" | "bid_sweep" | "ask_sweep"
    signal: str = "NEUTRAL"                   # "STRONG_BUY" | "BUY" | "NEUTRAL" | "SELL" | "STRONG_SELL"
    top3_bid_ratio: float = 0.0              # Rasio dominasi bid di 3 level teratas
    spread_bps: float = 0.0                  # Spread dalam basis point
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RegimeSignal(Model):
    """Output HMM/GMM market regime classifier."""
    pair: str
    regime: str                               # "trending_up" | "trending_down" | "sideways_low_vol" | "sideways_high_vol" | "breakout"
    state: int = 0                            # State index dari HMM
    confidence: float = 0.0                  # [0, 1]
    volatility_percentile: float = 0.0       # Persentil volatilitas saat ini (0-100)
    trend_strength: float = 0.0              # ADX-equivalent strength [0, 1]
    regime_duration_bars: int = 0            # Berapa lama regime ini sudah berlangsung
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MAEMFEPrediction(Model):
    """Prediksi optimal Stop-Loss dan Take-Profit dari ML (MAE/MFE predictor)."""
    pair: str
    side: str                                 # "BUY" | "SELL"
    entry_price: float
    stop_loss: float                          # Harga SL yang direkomendasikan
    take_profit: float                        # Harga TP yang direkomendasikan
    mae_pct: float                            # Prediksi MAE dalam persen
    mfe_pct: float                            # Prediksi MFE dalam persen
    sl_pct: float                             # SL jarak dari entry dalam persen
    tp_pct: float                             # TP jarak dari entry dalam persen
    risk_reward_ratio: float = 0.0           # TP / SL ratio
    source: str = "rule_based"               # "ml_model" | "rule_based"
    confidence: float = 0.5
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def model_post_init(self, __context: Any) -> None:
        if self.sl_pct > 0:
            self.risk_reward_ratio = round(self.tp_pct / self.sl_pct, 2)


__all__ = [
    "OrderSide", "OrderType", "OrderStatus", "TimeInForce", "MarginMode",
    "PositionSide", "RiskDecision", "KillSwitchLevel",
    "OrderStatusEnum", "KillSwitchLevelEnum", "RiskDecisionEnum",
    "Model", "Order", "Position", "Fill", "MarketCandle", "MarketSnapshot",
    "TradeIntent", "CheckResult", "RiskCheckResult", "HealthCheck",
    "Proposal", "Experiment", "KillSwitchState", "TradeDossier", "AuditEvent",
    "OBISnapshot", "RegimeSignal", "MAEMFEPrediction",
]
