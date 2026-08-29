import asyncio
import os
import uuid

import structlog
from sqlalchemy import select

from shared.db.models import Incident, Order, Position
from shared.db.session import AsyncSessionLocal, init_db
from shared.messaging import Channels, MessageBus
from shared.schemas import OrderStatusEnum
from shared.security import load_secrets_into_env

load_secrets_into_env()
logger = structlog.get_logger()

message_bus = MessageBus()
RECONCILE_INTERVAL = int(os.getenv("RECONCILE_INTERVAL_SECONDS", "30"))


async def reconcile_loop():
    logger.info("Starting reconciler loop", interval=RECONCILE_INTERVAL)
    while True:
        try:
            await reconcile_positions()
            await reconcile_orders()
        except Exception as e:
            logger.error("Reconciliation error", error=str(e))
        await asyncio.sleep(RECONCILE_INTERVAL)


async def reconcile_positions():
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Position).where(Position.size > 0)
        )
        local_positions = result.scalars().all()

        if not local_positions:
            return

        adapter_url = "http://binance-adapter:8000"
        import aiohttp
        async with aiohttp.ClientSession() as session:
            for pos in local_positions:
                try:
                    async with session.get(f"{adapter_url}/positions/{pos.pair}") as resp:
                        if resp.status == 200:
                            remote = await resp.json()
                            if not _positions_match(pos, remote):
                                await report_mismatch("position", pos.position_id, pos.pair)
                        else:
                            await report_mismatch("position_api_error", pos.position_id, pos.pair)
                except Exception as e:
                    logger.error("Position reconcile error", pair=pos.pair, error=str(e))
                    await report_mismatch("position_connection_error", pos.position_id, pos.pair)


async def reconcile_orders():
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Order).where(Order.status.in_([
                OrderStatusEnum.SUBMITTED, OrderStatusEnum.PARTIALLY_FILLED,
                OrderStatusEnum.FILLED, OrderStatusEnum.PROTECTION_PENDING
            ]))
        )
        local_orders = result.scalars().all()

        if not local_orders:
            return

        adapter_url = "http://binance-adapter:8000"
        import aiohttp
        async with aiohttp.ClientSession() as session:
            for order in local_orders:
                try:
                    async with session.get(
                        f"{adapter_url}/orders/{order.exchange_order_id}",
                        params={"pair": order.pair}
                    ) as resp:
                        if resp.status == 200:
                            remote = await resp.json()
                            if remote.get("status") != order.status.value:
                                logger.warning("Order status mismatch",
                                    order_id=order.client_order_id,
                                    local=order.status.value,
                                    remote=remote.get("status"))
                                await publish_event("order_mismatch", {
                                    "order_id": order.client_order_id,
                                    "trade_id": order.trade_id,
                                    "pair": order.pair,
                                    "local_status": order.status.value,
                                    "remote_status": remote.get("status"),
                                })
                        else:
                            await report_mismatch("order_api_error", order.client_order_id, order.pair)
                except Exception as e:
                    logger.error("Order reconcile error", order_id=order.client_order_id, error=str(e))
                    await report_mismatch("order_connection_error", order.client_order_id, order.pair)


from shared.quant.supreme_final import KalmanReconciler

_kalman_rec = KalmanReconciler(process_var=1e-5, measurement_var=1e-3)

def _positions_match(local: Position, remote: dict) -> bool:
    diff = float(local.size) - float(remote.get("size", 0))
    res = _kalman_rec.update(diff)
    return not res["is_anomaly"] and abs(diff) < 0.001


async def report_mismatch(mismatch_type: str, resource_id: str, pair: str):
    async with AsyncSessionLocal() as db:
        incident = Incident(
            incident_id=f"INC-{uuid.uuid4().hex[:8].upper()}",
            incident_type=mismatch_type,
            severity="red",
            title=f"Reconciliation mismatch: {mismatch_type}",
            description=f"Resource {resource_id} on {pair}",
            related_ids={"resource_id": resource_id, "pair": pair},
            status="open",
        )
        db.add(incident)
        await db.commit()

    await publish_event("reconciliation_mismatch", {
        "type": mismatch_type,
        "resource_id": resource_id,
        "pair": pair,
    })


async def publish_event(event_type: str, data: dict):
    await message_bus.publish(Channels.RECONCILIATION, {
        "event_type": event_type,
        "data": data,
    })


async def reconcile_on_startup():
    logger.info("Running startup reconciliation")
    await reconcile_positions()
    await reconcile_orders()
    logger.info("Startup reconciliation complete")


async def main():
    await init_db()
    await message_bus.connect()
    await reconcile_on_startup()
    await reconcile_loop()


if __name__ == "__main__":
    asyncio.run(main())
