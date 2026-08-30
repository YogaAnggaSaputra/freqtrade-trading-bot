import uuid
import os
import aiohttp
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from decimal import Decimal

import structlog
import uvicorn
from client import BinanceFuturesClient
from execution_profiler import ExecutionProfiler
from fastapi import FastAPI, HTTPException
from limit_chaser import LimitChaser
from pydantic import BaseModel

from shared.db.session import close_db, init_db
from shared.messaging import Channels, MessageBus
from shared.schemas import (
    HealthCheck,
    MarginMode,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    PositionSide,
    TimeInForce,
)
from shared.security import load_secrets_into_env
from shared.metrics import add_metrics_endpoint

load_secrets_into_env()
logger = structlog.get_logger()

message_bus = MessageBus()
client = BinanceFuturesClient()
profiler = ExecutionProfiler()          # Execution quality metrics tracker
chaser  = LimitChaser(                  # Limit Maker fee-saving engine
    binance_client=client,
    execution_profiler=profiler,
)
EXECUTION_GUARD_ENABLED = os.getenv("EXECUTION_GUARD_ENABLED", "false").lower() == "true"
EXECUTION_GUARD_URL = os.getenv("EXECUTION_GUARD_URL", "http://execution-guard:8000")


async def _validate_submission_with_execution_guard(
    *, pair: str, client_order_id: str, price: float, amount: float, leverage: int,
) -> None:
    """Last-mile validation before Binance submission; fail closed when enabled."""
    if not EXECUTION_GUARD_ENABLED:
        return
    try:
        market = await client.get_ticker(pair)
        mid_price = float(market.get("price", 0) or 0)
        if mid_price <= 0:
            raise HTTPException(status_code=503, detail="Exchange returned no usable market price")
        pending = await client.get_orders(pair)
        duplicate = any(str(item.get("clientOrderId", "")) == client_order_id for item in pending)
        usage = client.get_rate_limit_status()
        if usage.get("used_weight_1m", 0) >= int(os.getenv("BINANCE_WEIGHT_LIMIT", "1100")):
            raise HTTPException(status_code=503, detail="Binance request weight near limit")
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1.0)) as session:
            async with session.post(f"{EXECUTION_GUARD_URL}/validate", json={
                "pair": pair, "price": price or mid_price, "mid_price": mid_price,
                "amount": amount, "leverage": leverage,
                "pending_same_order": duplicate,
            }) as response:
                if response.status != 200:
                    raise HTTPException(status_code=503, detail="Execution guard unavailable")
                result = await response.json()
                if not result.get("approved", False):
                    raise HTTPException(status_code=409, detail={"execution_guard": result.get("reasons", [])})
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Execution guard error: {exc}") from exc


async def validate_with_execution_guard(req: "PlaceOrderRequest") -> None:
    """Last-mile validation for the normal order endpoint."""
    await _validate_submission_with_execution_guard(
        pair=req.pair,
        client_order_id=req.client_order_id,
        price=float(req.price or 0),
        amount=float(req.amount),
        leverage=req.leverage,
    )


class PlaceOrderRequest(BaseModel):
    trade_id: str
    client_order_id: str
    pair: str
    side: OrderSide
    order_type: OrderType
    amount: Decimal
    price: Decimal | None = None
    leverage: int
    margin_mode: MarginMode
    stop_loss: Decimal
    take_profit: Decimal | None = None
    time_in_force: TimeInForce = TimeInForce.GTC
    reduce_only: bool = False


class CancelOrderRequest(BaseModel):
    order_id: str
    pair: str


async def handle_kill_switch(payload: dict) -> None:
    """Close exchange positions for RED/BLACK kill-switch events."""
    level = str(payload.get("level", "")).lower()
    if level not in {"red", "black"} or os.getenv("KILL_SWITCH_CLOSE_POSITIONS", "false").lower() != "true":
        return
    positions = await client.get_positions()
    closed = 0
    for position in positions:
        amount = float(position.get("positionAmt", "0") or 0)
        if not amount:
            continue
        symbol = str(position.get("symbol", ""))
        side = "SELL" if amount > 0 else "BUY"
        await client.place_order(symbol=symbol, side=side, order_type="market",
                                 size=str(abs(amount)), leverage=1,
                                 margin_mode="isolated", reduce_only=True)
        closed += 1
    logger.critical("Kill switch %s closed %d exchange positions", level, closed)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    await message_bus.connect()
    await client.start()
    await message_bus.subscribe(Channels.KILL_SWITCH, handle_kill_switch)
    await message_bus.start_listening()
    yield
    await client.stop()
    await message_bus.disconnect()
    await close_db()


app = FastAPI(title="Binance Adapter", lifespan=lifespan)

add_metrics_endpoint(app)


@app.get("/health")
async def health():
    return HealthCheck(service="binance-adapter", status="healthy", checks={}, timestamp=datetime.now(UTC)).model_dump()


@app.get("/account")
async def get_account():
    """Kembalikan saldo akun Binance Futures (totalWalletBalance dalam USDT)."""
    try:
        result = await client.get_account()
        return {
            "totalWalletBalance": result.get("totalWalletBalance", "0"),
            "totalUnrealizedProfit": result.get("totalUnrealizedProfit", "0"),
            "totalMarginBalance": result.get("totalMarginBalance", "0"),
            "availableBalance": result.get("availableBalance", "0"),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Get account failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/orders")
async def place_order(req: PlaceOrderRequest):
    try:
        await validate_with_execution_guard(req)
        result = await client.place_order(
            symbol=req.pair,
            side=req.side.value,
            order_type=req.order_type.value,
            size=str(req.amount),
            price=str(req.price) if req.price else None,
            leverage=req.leverage,
            margin_mode=req.margin_mode.value,
            stop_loss=str(req.stop_loss),
            take_profit=str(req.take_profit) if req.take_profit else None,
            client_order_id=req.client_order_id,
            reduce_only=req.reduce_only,
        )

        order = Order(
            order_id=str(result.get("orderId", uuid.uuid4())),
            client_order_id=req.client_order_id,
            exchange_order_id=str(result.get("orderId")),
            trade_id=req.trade_id,
            pair=req.pair,
            side=req.side,
            order_type=req.order_type,
            status=OrderStatus.SUBMITTED,
            amount=req.amount,
            filled=Decimal("0"),
            price=req.price or Decimal("0"),
            leverage=req.leverage,
            margin_mode=req.margin_mode,
            time_in_force=req.time_in_force,
            stop_loss=req.stop_loss,
            take_profit=req.take_profit,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            raw_response=result,
        )

        await message_bus.publish(Channels.ORDER_UPDATE, {"order": order.model_dump()})
        return {"success": True, "order_id": order.order_id, "client_order_id": order.client_order_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Place order failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


class LimitChaserRequest(BaseModel):
    trade_id: str
    pair: str
    side: OrderSide               # BUY | SELL
    quantity: float
    intended_price: float
    tick_size: float
    regime: str | None = None
    adx: float | None = None
    client_order_id: str | None = None


@app.post("/orders/limit")
async def place_limit_order(req: LimitChaserRequest):
    """
    Place order via LimitChaser — otomatis pilih Limit Maker (fee ~60% lebih hemat)
    atau Market order tergantung kondisi pasar.
    """
    try:
        client_order_id = req.client_order_id or f"LIMIT-{req.trade_id}-{uuid.uuid4().hex[:12]}"
        await _validate_submission_with_execution_guard(
            pair=req.pair,
            client_order_id=client_order_id,
            price=req.intended_price,
            amount=req.quantity,
            leverage=1,
        )
        result = await chaser.execute(
            pair=req.pair,
            side=req.side.value,
            quantity=req.quantity,
            intended_price=req.intended_price,
            tick_size=req.tick_size,
            regime=req.regime,
            adx=req.adx,
            client_order_id=client_order_id,
            trade_id=req.trade_id,
        )
        if result.success:
            recent = profiler.get_recent_records(1)
            if recent and recent[0].get("is_alert"):
                await message_bus.publish(Channels.ALERT, {
                    "type": "execution_quality_alert",
                    "severity": "high",
                    "trade_id": req.trade_id,
                    "pair": req.pair,
                    "reasons": recent[0].get("alert_reasons", []),
                })
        return {
            "success": result.success,
            "order_id": result.order_id,
            "executed_price": result.executed_price,
            "executed_qty": result.executed_qty,
            "execution_mode": result.execution_mode,
            "repeg_count": result.repeg_count,
            "fill_type": result.fill_type,
            "slippage_bps": result.slippage_bps,
            "fee_savings_usdt": result.fee_savings_estimate_usdt,
            "error": result.error,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Limit chaser order failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/execution/summary")
async def execution_summary():
    """Rolling execution quality metrics for ops/dashboard consumers."""
    return profiler.get_summary()


@app.delete("/orders/{order_id}")
async def cancel_order(order_id: str, pair: str):
    try:
        result = await client.cancel_order(order_id, pair)
        return {"success": True, "result": result}
    except Exception as e:
        logger.error("Cancel order failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/orders/{order_id}")
async def get_order(order_id: str, pair: str):
    try:
        result = await client.get_order_detail(order_id, pair)
        return result
    except Exception as e:
        logger.error("Get order failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/positions/{pair}")
async def get_position(pair: str):
    try:
        result = await client.get_positions(pair)
        if result:
            pos_data = result[0]
            position = Position(
                position_id=str(pos_data.get("positionId", pair)),
                pair=pair,
                side=PositionSide.LONG if float(pos_data.get("positionAmt", "0")) > 0 else PositionSide.SHORT,
                size=abs(Decimal(str(pos_data.get("positionAmt", "0")))),
                entry_price=Decimal(str(pos_data.get("entryPrice", "0"))),
                mark_price=Decimal(str(pos_data.get("markPrice", "0"))),
                leverage=int(pos_data.get("leverage", "1")),
                margin_mode=MarginMode.ISOLATED if pos_data.get("marginType") == "isolated" else MarginMode.CROSSED,
                unrealized_pnl=Decimal(str(pos_data.get("unRealizedProfit", "0"))),
                realized_pnl=Decimal("0"),
                liquidation_price=Decimal(str(pos_data.get("liquidationPrice", "0"))) if pos_data.get("liquidationPrice") else None,
                margin_ratio=None,
                stop_loss=None,
                take_profit=None,
                opened_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
                exchange_position_id=str(pos_data.get("positionId", pair)),
            )
            return position.model_dump()
        return {"size": "0"}
    except Exception as e:
        logger.error("Get position failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/reconcile/orders")
async def reconcile_orders(pair: str):
    try:
        local_orders = await get_local_open_orders(pair)
        exchange_orders = await client.get_orders(pair)

        mismatches = []
        for local in local_orders:
            found = False
            for ex in exchange_orders:
                if str(ex.get("orderId")) == local.exchange_order_id or ex.get("clientOrderId") == local.client_order_id:
                    found = True
                    break
            if not found:
                mismatches.append(local)

        if mismatches:
            await message_bus.publish(Channels.RECONCILIATION, {
                "type": "orders",
                "pair": pair,
                "mismatches": [m.client_order_id for m in mismatches],
            })

        return {"mismatches": len(mismatches), "details": [m.client_order_id for m in mismatches]}
    except Exception as e:
        logger.error("Reconcile orders failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/reconcile/positions")
async def reconcile_positions(pair: str):
    try:
        local_pos = await get_local_position(pair)
        exchange_pos = await client.get_positions(pair)

        mismatch = False
        if local_pos and exchange_pos:
            ex = exchange_pos[0]
            if abs(local_pos.size - abs(Decimal(str(ex.get("positionAmt", "0"))))) > Decimal("0.0001"):
                mismatch = True

        if mismatch:
            await message_bus.publish(Channels.RECONCILIATION, {
                "type": "positions",
                "pair": pair,
                "local": local_pos.model_dump() if local_pos else None,
                "exchange": exchange_pos,
            })

        return {"mismatch": mismatch}
    except Exception as e:
        logger.error("Reconcile positions failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


async def get_local_open_orders(pair: str):
    from sqlalchemy import select

    from shared.db.models import Order as OrderModel
    from shared.db.models import OrderStatusEnum
    from shared.db.session import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(OrderModel).where(
                OrderModel.pair == pair,
                OrderModel.status.in_([
                    OrderStatusEnum.SUBMITTED,
                    OrderStatusEnum.PARTIALLY_FILLED,
                    OrderStatusEnum.PROTECTION_PENDING,
                ])
            )
        )
        return result.scalars().all()


async def get_local_position(pair: str):
    from sqlalchemy import select

    from shared.db.models import Position as PositionModel
    from shared.db.session import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(PositionModel).where(PositionModel.pair == pair)
        )
        return result.scalar_one_or_none()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
