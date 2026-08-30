from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum

import yaml
from pydantic import BaseModel, Field


class RiskDecisionEnum(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"


class CheckResult(BaseModel):
    name: str
    passed: bool
    reason: str
    details: dict = Field(default_factory=dict)


class RiskCheckResult(BaseModel):
    decision: RiskDecisionEnum
    reason: str
    checks: list[CheckResult]
    approved_size: Decimal | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PolicyConfig(BaseModel):
    trading_mode: str = "demo"
    margin_mode: str = "isolated"
    pair_allowlist: list[str] = Field(default_factory=lambda: ["BTCUSDT"])
    max_leverage: int = 3
    risk_per_trade_pct: float = 0.005
    max_total_exposure_pct: float = 0.10
    max_open_trades: int = 1
    daily_loss_limit_pct: float = 0.02
    max_drawdown_pct: float = 0.05
    stoploss_mandatory: bool = True
    position_adjustment: bool = False
    api_withdrawal_enabled: bool = False
    live_promotion_requires_approval: bool = True
    market_data_freshness_seconds: int = 60


def load_policy(path: str = "/config/policy.yaml") -> PolicyConfig:
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
        policy = PolicyConfig(**data)
    except Exception:
        policy = PolicyConfig()

    # Mode live/demo dikontrol via env TRADE_MODE — override trading_mode
    # agar konsisten antara freqtrade-runtime, risk-gateway, dan policy.
    # LIVE_MODE_OVERRIDE=true memungkinkan live meski policy.yaml bilang demo.
    import os
    trade_mode_env = os.getenv("TRADE_MODE", "").lower()
    if trade_mode_env == "live":
        policy.trading_mode = "live"
    elif trade_mode_env == "demo":
        policy.trading_mode = "demo"
    return policy
