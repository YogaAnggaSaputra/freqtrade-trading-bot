import os
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import aiohttp
import redis.asyncio as redis
from policy import CheckResult, PolicyConfig, RiskCheckResult
from spread_guard import SpreadGuard
from sqlalchemy import and_, func, select

from shared.db.models import Deployment, Order, Position, TradeDossier
from shared.db.session import AsyncSessionLocal
from shared.schemas import OrderStatusEnum, RiskDecisionEnum
from shared.security import get_secret

MARKET_DATA_GW_URL = os.getenv("MARKET_DATA_GATEWAY_URL", "http://market-data-gateway:8000")
MODEL_INFERENCE_URL = os.getenv("MODEL_INFERENCE_URL", "http://model-inference:8000")


class RiskGateway:
    def __init__(self, policy: PolicyConfig, macro_filter: Any | None = None):
        self.policy = policy
        self._redis: redis.Redis | None = None
        self.macro_filter = macro_filter  # MacroFilter instance (opsional)

    async def _get_redis(self) -> redis.Redis:
        if self._redis is None:
            self._redis = redis.Redis(
                host=os.getenv("REDIS_HOST", "redis"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                password=get_secret("redis_password"),
                decode_responses=True,
            )
        return self._redis

    async def _set_kill_switch(self, level: str) -> None:
        """Tulis level kill switch ke Redis (konsisten dengan yang dibaca check_kill_switch).

        Nilai 'green' / None / '' → hapus key (normal operation).
        """
        r = await self._get_redis()
        if not level or level.lower() in ("green", "none", ""):
            await r.delete("kill_switch:level")
        else:
            await r.set("kill_switch:level", level.lower())

    async def check_kill_switch(self) -> CheckResult:
        r = await self._get_redis()
        level = await r.get("kill_switch:level")
        if level:
            return CheckResult(
                name="kill_switch",
                passed=False,
                reason=f"Kill switch active: {level}",
                details={"level": level}
            )
        return CheckResult(name="kill_switch", passed=True, reason="No active kill switch")

    async def check_macro_event(self) -> CheckResult:
        """
        Cek apakah ada high-impact economic event yang sedang memblokir trading.
        Jika MacroFilter tidak terpasang, check ini selalu pass (fail-open).
        Event FOMC/CPI/Fed → window 2x lebih panjang dari event biasa.
        """
        if self.macro_filter is None:
            return CheckResult(name="macro_event", passed=True, reason="Macro filter not configured")

        is_blocked, reason, event_details = self.macro_filter.check_blocking()
        if is_blocked:
            return CheckResult(
                name="macro_event",
                passed=False,
                reason=reason,
                details=event_details,
            )
        return CheckResult(
            name="macro_event",
            passed=True,
            reason="No high-impact macro events in active block window",
        )

    async def check_environment(self, trade_mode: str) -> CheckResult:
        if trade_mode != self.policy.trading_mode:
            return CheckResult(
                name="environment",
                passed=False,
                reason=f"Trade mode mismatch: requested {trade_mode}, policy {self.policy.trading_mode}"
            )
        return CheckResult(name="environment", passed=True, reason=f"Mode: {trade_mode}")

    async def check_pair_allowlist(self, pair: str) -> CheckResult:
        # Normalisasi: "DYDX/USDT:USDT" → "DYDXUSDT" (sama dengan freshness key)
        if "/" in pair:
            base, rest = pair.split("/", 1)
            quote = rest.split(":", 1)[0] if ":" in rest else rest
            normalized = (base + quote).upper().replace(":", "")
        else:
            normalized = pair.upper().replace(":", "")
        if normalized not in self.policy.pair_allowlist:
            return CheckResult(
                name="pair_allowlist",
                passed=False,
                reason=f"Pair {pair} not in allowlist: {self.policy.pair_allowlist}"
            )
        return CheckResult(name="pair_allowlist", passed=True, reason=f"Pair {pair} allowed")

    async def check_reconciliation(self, pair: str) -> CheckResult:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(Order).where(
                    and_(Order.pair == pair, Order.status.in_([
                        OrderStatusEnum.SUBMITTED, OrderStatusEnum.PARTIALLY_FILLED,
                        OrderStatusEnum.FILLED, OrderStatusEnum.PROTECTION_PENDING
                    ]))
                )
            )
            local_orders = result.scalars().all()

            if local_orders:
                return CheckResult(
                    name="reconciliation",
                    passed=False,
                    reason=f"Unreconciled orders exist for {pair}",
                    details={"count": len(local_orders)}
                )
        return CheckResult(name="reconciliation", passed=True, reason="No pending orders")

    async def check_market_data_freshness(self, pair: str, max_age_seconds: int | None = None) -> CheckResult:
        if max_age_seconds is None:
            max_age_seconds = self.policy.market_data_freshness_seconds
        r = await self._get_redis()
        # Normalisasi pair: "DYDX/USDT:USDT" → "DYDXUSDT" (format key di Redis
        # ditulis oleh market-data-gateway dari symbol Binance "DYDXUSDT").
        # Tanpa normalisasi, key yang dicari = "DYDX/USDT:USDT" → mismatch → FAIL CLOSED.
        if "/" in pair:
            base, rest = pair.split("/", 1)
            quote = rest.split(":", 1)[0] if ":" in rest else rest
            normalized = (base + quote).upper().replace(":", "")
        else:
            normalized = pair.upper().replace(":", "")
        last_update = await r.get(f"market:last_update:{normalized}")
        if not last_update:
            return CheckResult(
                name="market_data_freshness",
                passed=False,
                reason=f"No market data for {pair}"
            )

        try:
            last_ts = datetime.fromisoformat(last_update)
        except ValueError:
            return CheckResult(
                name="market_data_freshness",
                passed=False,
                reason=f"Invalid market data timestamp: {last_update}"
            )
        # Normalisasi: treat naive timestamp sebagai UTC agar bisa dibandingkan
        # dengan datetime.now(UTC) (timezone-aware).
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=UTC)

        age = (datetime.now(UTC) - last_ts).total_seconds()
        if age > max_age_seconds:
            return CheckResult(
                name="market_data_freshness",
                passed=False,
                reason=f"Market data stale: {age:.0f}s > {max_age_seconds}s",
                details={"age_seconds": age}
            )
        return CheckResult(name="market_data_freshness", passed=True, reason=f"Data fresh: {age:.1f}s")

    async def check_critical_alerts(self) -> CheckResult:
        r = await self._get_redis()
        critical = await r.get("alert:critical")
        if critical:
            return CheckResult(
                name="critical_alerts",
                passed=False,
                reason=f"Critical alert active: {critical}"
            )
        return CheckResult(name="critical_alerts", passed=True, reason="No critical alerts")

    async def check_stop_loss_valid(self, stop_loss: Decimal, entry_price: Decimal, side: str) -> CheckResult:
        if not self.policy.stoploss_mandatory:
            return CheckResult(name="stop_loss", passed=True, reason="SL not mandatory per policy")

        if stop_loss <= 0:
            return CheckResult(
                name="stop_loss",
                passed=False,
                reason="Stop loss must be > 0"
            )

        if side == "buy":
            if stop_loss >= entry_price:
                return CheckResult(
                    name="stop_loss",
                    passed=False,
                    reason="Long stop loss must be below entry price"
                )
        else:
            if stop_loss <= entry_price:
                return CheckResult(
                    name="stop_loss",
                    passed=False,
                    reason="Short stop loss must be above entry price"
                )
        return CheckResult(name="stop_loss", passed=True, reason="Stop loss valid")

    async def check_risk_per_trade(self, equity: Decimal, risk_amount: Decimal) -> CheckResult:
        max_risk = equity * Decimal(str(self.policy.risk_per_trade_pct))
        if risk_amount > max_risk:
            return CheckResult(
                name="risk_per_trade",
                passed=False,
                reason=f"Risk {risk_amount} exceeds limit {max_risk} ({self.policy.risk_per_trade_pct*100}% equity)"
            )
        return CheckResult(name="risk_per_trade", passed=True, reason=f"Risk {risk_amount} within limit {max_risk}")

    async def check_leverage(self, leverage: int) -> CheckResult:
        if leverage > self.policy.max_leverage:
            return CheckResult(
                name="leverage",
                passed=False,
                reason=f"Leverage {leverage}x exceeds max {self.policy.max_leverage}x"
            )
        return CheckResult(name="leverage", passed=True, reason=f"Leverage {leverage}x within limit")

    async def check_exposure(self, current_exposure: Decimal, new_notional: Decimal, equity: Decimal) -> CheckResult:
        max_exposure = equity * Decimal(str(self.policy.max_total_exposure_pct))
        projected = current_exposure + new_notional
        if projected > max_exposure:
            return CheckResult(
                name="exposure",
                passed=False,
                reason=f"Projected exposure {projected} exceeds limit {max_exposure}"
            )
        return CheckResult(name="exposure", passed=True, reason=f"Exposure {projected} within limit {max_exposure}")

    async def check_obi(self, pair: str, side: str) -> CheckResult:
        """
        Cek Order Book Imbalance (OBI) dari market-data-gateway.
        Blokir trade jika ada liquidity sweep berlawanan arah atau OBI ekstrem.
        """
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2)) as session:
                async with session.get(f"{MARKET_DATA_GW_URL}/orderbook/{pair}") as resp:
                    if resp.status != 200:
                        # Tidak bisa ambil OBI → tidak blokir (fail open)
                        return CheckResult(name="obi", passed=True, reason="OBI data unavailable — skipped")
                    data = await resp.json()

            obi = float(data.get("obi", 0.0))
            sweep = data.get("liquidity_sweep", "none")
            signal = data.get("signal", "NEUTRAL")
            side_upper = str(side).upper()

            # Blokir jika liquidity sweep berlawanan arah
            if side_upper == "BUY" and sweep == "ask_sweep":
                return CheckResult(
                    name="obi",
                    passed=False,
                    reason="Liquidity ask_sweep detected — institutional selling, blocking BUY",
                    details={"obi": obi, "sweep": sweep, "signal": signal},
                )
            if side_upper == "SELL" and sweep == "bid_sweep":
                return CheckResult(
                    name="obi",
                    passed=False,
                    reason="Liquidity bid_sweep detected — institutional buying, blocking SELL",
                    details={"obi": obi, "sweep": sweep, "signal": signal},
                )

            # Blokir jika OBI ekstrem berlawanan arah (>0.65)
            if side_upper == "BUY" and obi < -0.65:
                return CheckResult(
                    name="obi",
                    passed=False,
                    reason=f"OBI {obi:.3f} strongly negative — order book heavily skewed SELL, blocking BUY",
                    details={"obi": obi, "signal": signal},
                )
            if side_upper == "SELL" and obi > 0.65:
                return CheckResult(
                    name="obi",
                    passed=False,
                    reason=f"OBI {obi:.3f} strongly positive — order book heavily skewed BUY, blocking SELL",
                    details={"obi": obi, "signal": signal},
                )

            return CheckResult(
                name="obi",
                passed=True,
                reason=f"OBI {obi:.3f} acceptable for {side_upper} — signal: {signal}",
                details={"obi": obi, "sweep": sweep, "signal": signal},
            )
        except Exception as e:
            # Fail open: jika tidak bisa cek OBI, jangan blokir trade
            return CheckResult(name="obi", passed=True, reason=f"OBI check skipped: {e}")

    async def check_spread_guard(self, pair: str) -> CheckResult:
        """
        Guard real-time untuk bid-ask spread abnormal dan ATR velocity surge.
        Memblokir entry saat kondisi pasar berbahaya (flash crash, liquidity grab).
        Fail-open: jika data tidak tersedia, pass.
        """
        result = await SpreadGuard.check(pair)
        if not result.passed:
            return CheckResult(
                name="spread_guard",
                passed=False,
                reason=result.reason,
                details=result.details,
            )
        return CheckResult(
            name="spread_guard",
            passed=True,
            reason=result.reason,
            details=result.details,
        )

    async def check_daily_loss(self, equity: Decimal | None = None) -> CheckResult:
        """Cek daily loss pct. equity = total wallet balance dari Binance (atau fallback)."""
        import os
        base_equity = equity or Decimal(os.getenv("EQUITY_FALLBACK_USDT", "10000"))
        async with AsyncSessionLocal() as db:
            today = datetime.now(UTC).date()
            # Kolom closed_at di DB bertipe `timestamp without time zone` (naive).
            # Jangan set tzinfo — asyncpg gagal encode aware datetime ke kolom
            # naive (DataError: can't subtract offset-naive and offset-aware).
            start = datetime.combine(today, datetime.min.time())

            result = await db.execute(
                select(func.sum(TradeDossier.realized_pnl)).where(
                    and_(
                        TradeDossier.closed_at >= start,
                        TradeDossier.realized_pnl < 0
                    )
                )
            )
            daily_loss = result.scalar() or Decimal("0")

            # starting_equity = saldo real + akumulasi PnL sebelum hari ini
            equity_result = await db.execute(
                select(func.coalesce(func.sum(TradeDossier.realized_pnl), 0)).where(
                    TradeDossier.closed_at < start
                )
            )
            starting_equity = base_equity + (equity_result.scalar() or Decimal("0"))

            loss_pct = abs(daily_loss) / starting_equity if starting_equity > 0 else Decimal("0")
            if loss_pct >= Decimal(str(self.policy.daily_loss_limit_pct)):
                return CheckResult(
                    name="daily_loss",
                    passed=False,
                    reason=f"Daily loss {loss_pct*100:.2f}% exceeds limit {self.policy.daily_loss_limit_pct*100}%",
                    details={"daily_loss": str(daily_loss), "limit_pct": self.policy.daily_loss_limit_pct}
                )
        return CheckResult(name="daily_loss", passed=True, reason="Daily loss within limit")

    async def check_max_drawdown(self, equity: Decimal | None = None) -> CheckResult:
        """Cek max drawdown sepanjang history. equity = starting balance real."""
        import os
        base_equity = equity or Decimal(os.getenv("EQUITY_FALLBACK_USDT", "10000"))
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(TradeDossier).order_by(TradeDossier.created_at)
            )
            trades = result.scalars().all()

            if not trades:
                return CheckResult(name="max_drawdown", passed=True, reason="No trades yet")

            running_equity = base_equity
            peak = running_equity
            max_dd = Decimal("0")

            for trade in trades:
                running_equity += trade.realized_pnl
                if running_equity > peak:
                    peak = running_equity
                dd = (peak - running_equity) / peak if peak > 0 else Decimal("0")
                if dd > max_dd:
                    max_dd = dd

            if max_dd >= Decimal(str(self.policy.max_drawdown_pct)):
                return CheckResult(
                    name="max_drawdown",
                    passed=False,
                    reason=f"Max drawdown {max_dd*100:.2f}% exceeds limit {self.policy.max_drawdown_pct*100}%"
                )
        return CheckResult(name="max_drawdown", passed=True, reason=f"Max DD {max_dd*100:.2f}% within limit")

    async def check_strategy_approved(self, strategy_version: str) -> CheckResult:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(Deployment).where(
                    and_(
                        Deployment.strategy_version == strategy_version,
                        Deployment.status == "active"
                    )
                )
            )
            deployment = result.scalar_one_or_none()
            if not deployment:
                return CheckResult(
                    name="strategy_approved",
                    passed=False,
                    reason=f"Strategy {strategy_version} not deployed/active"
                )
        return CheckResult(name="strategy_approved", passed=True, reason="Strategy active")

    async def check_config_consistency(self, strategy_version: str, config_version: str) -> CheckResult:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(Deployment).where(
                    and_(
                        Deployment.strategy_version == strategy_version,
                        Deployment.config_version == config_version,
                        Deployment.status == "active"
                    )
                )
            )
            deployment = result.scalar_one_or_none()
            if not deployment:
                return CheckResult(
                    name="config_consistency",
                    passed=False,
                    reason=f"Config {config_version} not matching active deployment for {strategy_version}"
                )
        return CheckResult(name="config_consistency", passed=True, reason="Config matches deployment")

    async def validate_trade(
        self,
        trade_id: str,
        client_order_id: str,
        strategy_version: str,
        config_version: str,
        pair: str,
        side: str,
        order_type: str,
        amount: Decimal,
        price: Decimal | None,
        leverage: int,
        margin_mode: str,
        stop_loss: Decimal,
        take_profit: Decimal | None,
        timeframe: str,
        equity: Decimal,
        trade_mode: str = "demo",
    ) -> RiskCheckResult:
        checks: list[CheckResult] = []

        checks.append(await self.check_kill_switch())
        checks.append(await self.check_environment(trade_mode))
        checks.append(await self.check_pair_allowlist(pair))
        checks.append(await self.check_reconciliation(pair))
        checks.append(await self.check_market_data_freshness(pair))
        checks.append(await self.check_critical_alerts())

        entry_price = price if price else Decimal("0")
        checks.append(await self.check_stop_loss_valid(stop_loss, entry_price, side))
        # Risk aktual = jarak SL ke entry × qty (bukan notional penuh × pct).
        # Ini benar secara finansial: yang dipertaruhkan adalah jarak ke SL,
        # bukan seluruh notional posisi.
        if entry_price > 0 and stop_loss > 0:
            sl_distance = abs(entry_price - stop_loss)
            actual_risk = sl_distance * amount
        else:
            actual_risk = Decimal("0")
        checks.append(await self.check_risk_per_trade(equity, actual_risk))
        checks.append(await self.check_leverage(leverage))

        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(func.coalesce(func.sum(Position.size * Position.entry_price), 0)).where(
                    Position.size != 0
                )
            )
            current_exposure = result.scalar() or Decimal("0")
        # Exposure dihitung dari MARGIN (notional ÷ leverage), bukan notional penuh.
        # Ini benar: yang terkunci dari balance adalah margin, bukan nilai posisi.
        # Dengan micro-account ($1-2) dan leverage 5x, order $5 notional = $1 margin.
        leverage_safe = leverage if leverage > 0 else 1
        new_margin = amount * entry_price / Decimal(leverage_safe) if entry_price > 0 else Decimal("0")
        checks.append(await self.check_exposure(current_exposure, new_margin, equity))

        checks.append(await self.check_daily_loss(equity=equity))
        checks.append(await self.check_max_drawdown(equity=equity))
        checks.append(await self.check_obi(pair=pair, side=side))  # OBI liquidity check
        checks.append(await self.check_macro_event())              # Macro news calendar block
        checks.append(await self.check_spread_guard(pair=pair))    # Spread & volatility guard
        checks.append(await self.check_strategy_approved(strategy_version))
        checks.append(await self.check_config_consistency(strategy_version, config_version))

        failed = [c for c in checks if not c.passed]
        decision = RiskDecisionEnum.REJECTED if failed else RiskDecisionEnum.APPROVED
        reason = "; ".join([c.reason for c in failed]) if failed else "All checks passed"

        return RiskCheckResult(
            decision=decision,
            reason=reason,
            checks=checks,
            approved_size=amount if decision == RiskDecisionEnum.APPROVED else None,
        )
