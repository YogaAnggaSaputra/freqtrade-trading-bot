import os
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

message_bus = MessageBus()
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    await message_bus.connect()
    await macro_filter.start()  # Start economic calendar refresh
    yield
    await macro_filter.stop()
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
async def set_kill_switch(level: str, reason: str = "Set via API"):
    """Set/reset kill switch level yang dibaca engine saat validasi trade.

    level: yellow | orange | red | black | green (green = resume/normal).
    Ini adalah sumber kebenaran yang sama dengan yang di-set discord-bot.
    """
    valid = {"yellow", "orange", "red", "black", "green"}
    if level.lower() not in valid:
        raise HTTPException(status_code=400, detail=f"Invalid level: {level} (use {sorted(valid)})")
    await risk_gateway._set_kill_switch(level)
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
