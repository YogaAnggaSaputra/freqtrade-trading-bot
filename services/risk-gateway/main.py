import os
import asyncio
import logging
from contextlib import asynccontextmanager
from decimal import Decimal

import aiohttp
import uvicorn
from engine import RiskGateway
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from macro_filter import MacroFilter
from policy import load_policy
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from shared.db.session import close_db, init_db
from shared.messaging import Channels, MessageBus
from shared.schemas import TradeIntent
from shared.security import load_secrets_into_env

load_secrets_into_env()
logger = logging.getLogger("risk_gateway.main")

message_bus = MessageBus()
_dead_man_task: asyncio.Task | None = None
policy = load_policy(os.getenv("POLICY_PATH", "/config/policy.yaml"))
macro_filter = MacroFilter()               # Economic event calendar filter
risk_gateway = RiskGateway(policy, macro_filter=macro_filter)

BINANCE_ADAPTER_URL = os.getenv("BINANCE_ADAPTER_URL", "http://binance-adapter:8000")
EQUITY_FALLBACK = Decimal(os.getenv("EQUITY_FALLBACK_USDT", "10000"))

# Prometheus metrics
trade_validations_total = Counter(
    "risk_trade_validations_total", "Total trade validations", ["decision"]
)
validation_duration_seconds = Histogram(
    "risk_validation_duration_seconds", "Time spent on trade validation"
)
current_equity_gauge = Gauge(
    "risk_current_equity_usdt", "Current wallet equity from Binance"
)
ANOMALY_AUTO_PAUSE = os.getenv("ANOMALY_AUTO_PAUSE", "true").lower() == "true"
EXECUTION_AUTO_PAUSE = os.getenv("EXECUTION_AUTO_PAUSE", "false").lower() == "true"
DEAD_MAN_ENABLED = os.getenv("DEAD_MAN_ENABLED", "false").lower() == "true"
DEAD_MAN_TIMEOUT_SECONDS = int(os.getenv("DEAD_MAN_TIMEOUT_SECONDS", "1800"))


async def fetch_equity() -> Decimal:
    """Ambil totalWalletBalance dari binance-adapter. Fallback ke EQUITY_FALLBACK jika gagal."""
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
            async with session.get(f"{BINANCE_ADAPTER_URL}/account") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    balance = data.get("totalWalletBalance", "0")
                    return Decimal(str(balance))
    except Exception:  # noqa: BLE001
        pass
    return EQUITY_FALLBACK


async def handle_market_anomaly(payload: dict) -> None:
    """Freeze new entries briefly when the market feed reports bad data."""
    if not ANOMALY_AUTO_PAUSE or payload.get("severity") != "high":
        return
    reason = "market_feed_anomaly:" + ",".join(payload.get("reasons", []))
    await risk_gateway._set_kill_switch("orange")
    await message_bus.publish(Channels.KILL_SWITCH, {
        "level": "orange", "reason": reason,
        "close_positions": False, "activated_by": "anomaly-detection",
    })


async def handle_execution_alert(payload: dict) -> None:
    if not EXECUTION_AUTO_PAUSE or payload.get("type") != "execution_quality_alert":
        return
    await risk_gateway._set_kill_switch("orange")
    await message_bus.publish(Channels.KILL_SWITCH, {
        "level": "orange", "reason": "execution_quality:" + ",".join(payload.get("reasons", [])),
        "close_positions": False, "activated_by": "execution-profiler",
    })


async def _dead_man_loop() -> None:
    """Alert and freeze new entries if the market feed stops reporting."""
    import redis.asyncio as aioredis
    from datetime import datetime

    redis_client = aioredis.Redis(
        host=os.getenv("REDIS_HOST", "redis"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        password=os.getenv("REDIS_PASSWORD", "changeme"),
        decode_responses=True,
    )
    key = "market:last_update:" + os.getenv("DEAD_MAN_SYMBOL", "BTCUSDT").upper().replace("/", "").replace(":", "")
    alerted = False
    try:
        while True:
            try:
                raw = await redis_client.get(key)
                age = float("inf")
                if raw:
                    stamp = datetime.fromisoformat(str(raw))
                    age = (datetime.utcnow() - stamp.replace(tzinfo=None)).total_seconds()
                if age > DEAD_MAN_TIMEOUT_SECONDS:
                    if not alerted:
                        alerted = True
                        await risk_gateway._set_kill_switch("orange")
                        await message_bus.publish(Channels.ALERT, {
                            "type": "dead_man_market_feed_stale", "severity": "high",
                            "key": key, "age_seconds": age,
                        })
                elif alerted:
                    alerted = False
                    await message_bus.publish(Channels.ALERT, {
                        "type": "dead_man_market_feed_recovered", "severity": "info", "key": key,
                    })
            except Exception as exc:
                logger.warning("Dead-man monitor check failed: %s", exc)
            await asyncio.sleep(60)
    except asyncio.CancelledError:
        raise
    finally:
        await redis_client.aclose()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    await message_bus.connect()
    await macro_filter.start()  # Start economic calendar refresh
    await message_bus.subscribe(Channels.MARKET_ANOMALY, handle_market_anomaly)
    await message_bus.subscribe(Channels.ALERT, handle_execution_alert)
    await message_bus.start_listening()
    global _dead_man_task
    if DEAD_MAN_ENABLED:
        _dead_man_task = asyncio.create_task(_dead_man_loop())
    yield
    await macro_filter.stop()
    if _dead_man_task and not _dead_man_task.done():
        _dead_man_task.cancel()
        await asyncio.gather(_dead_man_task, return_exceptions=True)
    await message_bus.disconnect()
    await close_db()

app = FastAPI(title="Risk Gateway", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "risk-gateway"}


@app.get("/policy")
async def get_policy():
    return policy.model_dump()


@app.post("/validate", response_model=dict)
async def validate_trade(trade: TradeIntent):
    result = await risk_gateway.validate_trade(
        trade_id=trade.trade_id,
        client_order_id=trade.client_order_id,
        strategy_version=trade.strategy_version,
        config_version=trade.config_version,
        pair=trade.pair,
        side=trade.side.value,
        order_type=trade.order_type.value,
        amount=trade.amount,
        price=trade.price,
        leverage=trade.leverage,
        margin_mode=trade.margin_mode.value,
        stop_loss=trade.stop_loss,
        take_profit=trade.take_profit,
        timeframe=trade.timeframe,
        equity=await fetch_equity(),
        trade_mode=os.getenv("TRADE_MODE", "demo"),
    )

    trade_validations_total.labels(decision=result.decision.value).inc()

    await message_bus.publish(Channels.RISK_DECISION, {
        "trade_id": trade.trade_id,
        "client_order_id": trade.client_order_id,
        "decision": result.decision.value,
        "reason": result.reason,
    })

    return result.model_dump()


@app.get("/checks")
async def get_checks():
    """Run all checks independently and return results"""
    equity = await fetch_equity()
    current_equity_gauge.set(float(equity))
    checks = [
        await risk_gateway.check_kill_switch(),
        await risk_gateway.check_environment(os.getenv("TRADE_MODE", "demo")),
        await risk_gateway.check_critical_alerts(),
        await risk_gateway.check_daily_loss(equity=equity),
        await risk_gateway.check_max_drawdown(equity=equity),
    ]
    return {"checks": [c.model_dump() for c in checks]}


@app.post("/killswitch")
async def set_kill_switch(level: str, reason: str = "Set via API", pin: str | None = None):
    """Set/reset kill switch level yang dibaca engine saat validasi trade.

    level: yellow | orange | red | black | green (green = resume/normal).
    Ini adalah sumber kebenaran yang sama dengan yang di-set discord-bot.
    """
    valid = {"yellow", "orange", "red", "black", "green"}
    if level.lower() not in valid:
        raise HTTPException(status_code=400, detail=f"Invalid level: {level} (use {sorted(valid)})")
    expected_pin = os.getenv("KILL_SWITCH_PIN", "")
    if expected_pin and pin != expected_pin:
        raise HTTPException(status_code=403, detail="Invalid kill-switch PIN")
    await risk_gateway._set_kill_switch(level)
    await message_bus.publish(Channels.KILL_SWITCH, {
        "level": level.lower(), "reason": reason, "close_positions": level.lower() in {"red", "black"},
        "activated_by": "risk-gateway",
    })
    return {"level": level.lower(), "reason": reason, "status": "ok"}


@app.get("/metrics")
async def metrics():
    """Endpoint Prometheus scraping."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ─── Macro Event Endpoints ────────────────────────────────────────────────────

@app.get("/macro/status")
async def macro_status():
    """
    Status MacroFilter saat ini: apakah trading sedang diblokir karena
    event ekonomi high-impact (Fed CPI, FOMC, NFP, dll).
    """
    return macro_filter.get_status()


@app.get("/macro/events")
async def macro_upcoming_events(hours: int = 24):
    """
    Daftar high-impact economic events yang akan datang dalam N jam ke depan.
    Default: 24 jam. Gunakan ?hours=48 untuk 2 hari ke depan.
    """
    events = macro_filter.get_upcoming_events(hours=hours)
    return {
        "upcoming_events": events,
        "count": len(events),
        "window_hours": hours,
        "block_before_min": int(os.getenv("MACRO_BLOCK_MINUTES_BEFORE", "30")),
        "block_after_min": int(os.getenv("MACRO_BLOCK_MINUTES_AFTER", "60")),
    }


# ─── Kelly Sizing & Spread Guard Endpoints ────────────────────────────────────

@app.get("/kelly/size")
async def kelly_position_size():
    """
    Hitung ukuran posisi optimal menggunakan Fractional Kelly Criterion.
    Berdasarkan statistik win rate & avg win/loss dari 30 trade terakhir.
    """
    from kelly_sizer import KellySizer
    equity = await fetch_equity()
    result = await KellySizer.compute(equity)
    return {
        "equity_usdt": float(equity),
        "recommended_size_pct": result.size_pct,
        "recommended_size_usdt": result.size_usdt,
        "full_kelly_pct": result.full_kelly_pct,
        "win_rate": result.win_rate,
        "avg_win_pct": result.avg_win_pct,
        "avg_loss_pct": result.avg_loss_pct,
        "risk_reward_ratio": result.risk_reward_ratio,
        "trade_count": result.trade_count,
        "mode": result.mode,
        "reason": result.reason,
    }


@app.get("/spread-guard/{pair}")
async def spread_guard_check(pair: str):
    """
    Jalankan Spread Guard & Volatility Guard untuk pair tertentu.
    Menunjukkan apakah kondisi pasar saat ini aman untuk entry.
    """
    from spread_guard import SpreadGuard
    result = await SpreadGuard.check(pair.upper())
    return {
        "pair": pair.upper(),
        "safe_to_trade": result.passed,
        "spread_blocked": result.spread_blocked,
        "volatility_blocked": result.volatility_blocked,
        "spread_z_score": result.spread_z_score,
        "atr_surge_pct": result.atr_surge_pct,
        "reason": result.reason,
        "details": result.details,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
