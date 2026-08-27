"""
main.py — Loss Analyzer Service
=================================
FastAPI service untuk analisis loss pattern dan drift detection.
Berjalan secara periodik dan menyediakan REST API untuk query hasil analisis.
"""
import asyncio
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import structlog
import uvicorn
from analyzer import LossAnalyzer
from fastapi import FastAPI, Query

from shared.db.session import close_db, init_db
from shared.messaging import Channels, MessageBus
from shared.schemas import HealthCheck
from shared.security import load_secrets_into_env
from shared.metrics import add_metrics_endpoint

load_secrets_into_env()
logger = structlog.get_logger("loss_analyzer.main")

message_bus = MessageBus()
analyzer = LossAnalyzer()
_background_task: asyncio.Task | None = None
_sweep_task: asyncio.Task | None = None

ANALYSIS_INTERVAL = int(os.getenv("LOSS_ANALYSIS_INTERVAL_HOURS", "4")) * 3600
RECONCILIATION_INTERVAL = int(os.getenv("FB_RECONCILIATION_INTERVAL_S", "900"))  # 15 menit

# ── Fase 2: Attribution engine ────────────────────────────────────────────
from attribution import AttributionEngine  # noqa: E402
attribution_engine = AttributionEngine(message_bus=message_bus)


async def _periodic_analysis():
    """Background loop: klasifikasi + pattern detection periodik."""
    while True:
        try:
            logger.info("Running periodic loss analysis...")
            classification = await analyzer.run_batch_classification(days=7)
            patterns = await analyzer.detect_systematic_patterns(days=14)

            for pattern in patterns:
                strategy_version = os.getenv("ACTIVE_STRATEGY_VERSION", "AITradingStrategy")
                incident_id = await analyzer.create_incident_if_needed(pattern, strategy_version)
                if incident_id:
                    await message_bus.publish(
                        Channels.ALERT,
                        {
                            "type": "loss_pattern_detected",
                            "pattern": pattern["pattern"],
                            "severity": pattern["severity"],
                            "incident_id": incident_id,
                        },
                    )

            logger.info(
                "Loss analysis complete",
                classified=classification.get("classified_count", 0),
                patterns=len(patterns),
            )
        except Exception as e:  # noqa: BLE001
            logger.error("Periodic analysis failed", error=str(e))
        await asyncio.sleep(ANALYSIS_INTERVAL)


async def _periodic_sweep():
    """Fase 2: reconciliation sweep tiap 15 menit untuk trade_outcomes
    yang terlewat (Redis at-most-once)."""
    while True:
        try:
            n = await attribution_engine.reconciliation_sweep()
            if n:
                logger.info("reconciliation sweep processed %d trades", n)
        except Exception as e:  # noqa: BLE001
            logger.warning("reconciliation sweep failed: %s", e)
        await asyncio.sleep(RECONCILIATION_INTERVAL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _background_task, _sweep_task
    await init_db()
    await message_bus.connect()

    # Fase 2: subscribe Channels.TRADE_CLOSED → attribution
    await message_bus.subscribe(Channels.TRADE_CLOSED, attribution_engine.process_trade)
    await message_bus.start_listening()
    logger.info("Subscribed to Channels.TRADE_CLOSED")

    _background_task = asyncio.create_task(_periodic_analysis())
    _sweep_task = asyncio.create_task(_periodic_sweep())
    logger.info("Loss Analyzer started")
    yield
    for task in (_background_task, _sweep_task):
        if task and not task.done():
            task.cancel()
    await message_bus.disconnect()
    await close_db()


app = FastAPI(
    title="Loss Analyzer",
    description="Deteksi pola loss sistematis, drift pasar, dan kegagalan teknis.",
    version="1.0.0",
    lifespan=lifespan,
)

add_metrics_endpoint(app)


@app.get("/health")
async def health():
    return HealthCheck(
        service="loss-analyzer",
        status="healthy",
        checks={"message_bus": message_bus.connected},
        timestamp=datetime.now(UTC),
    ).model_dump()


@app.post("/classify", response_model=dict[str, Any])
async def classify_losses(
    days: int = Query(default=7, ge=1, le=90),
    strategy_version: str | None = None,
):
    """Klasifikasikan semua trade loss yang belum terklasifikasi."""
    result = await analyzer.run_batch_classification(
        days=days, strategy_version=strategy_version
    )
    return result


@app.get("/patterns", response_model=dict[str, Any])
async def get_patterns(
    days: int = Query(default=14, ge=1, le=90),
    strategy_version: str | None = None,
    min_samples: int = Query(default=10, ge=5),
):
    """Deteksi pola loss sistematis."""
    patterns = await analyzer.detect_systematic_patterns(
        days=days,
        strategy_version=strategy_version,
        min_samples=min_samples,
    )
    return {"patterns": patterns, "count": len(patterns)}


@app.get("/drift/{pair}", response_model=dict[str, Any])
async def get_drift_score(pair: str, days: int = Query(default=7, ge=1, le=30)):
    """Hitung market drift score untuk pair tertentu."""
    result = await analyzer.compute_drift_score(pair=pair, days=days)
    return result


@app.get("/summary", response_model=dict[str, Any])
async def get_analysis_summary(
    days: int = Query(default=14, ge=1, le=90),
    strategy_version: str | None = None,
):
    """Ringkasan lengkap analisis loss."""
    from datetime import timedelta

    from sqlalchemy import and_, func, select

    from shared.db.models import TradeDossier
    from shared.db.session import AsyncSessionLocal

    since = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=days)

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(
                TradeDossier.loss_classification,
                func.count().label("count"),
            ).where(
                and_(
                    TradeDossier.created_at >= since,
                    TradeDossier.realized_pnl < 0,
                    TradeDossier.loss_classification.isnot(None),
                )
            ).group_by(TradeDossier.loss_classification)
        )
        classification_summary = {row[0]: row[1] for row in result.all()}

    patterns = await analyzer.detect_systematic_patterns(
        days=days, strategy_version=strategy_version
    )

    return {
        "lookback_days": days,
        "strategy_version": strategy_version,
        "classification_breakdown": classification_summary,
        "total_classified": sum(classification_summary.values()),
        "patterns_detected": len(patterns),
        "patterns": patterns,
        "timestamp": datetime.now(UTC).isoformat(),
    }


@app.post("/attribution/sweep", response_model=dict[str, Any])
async def trigger_sweep():
    """Manual trigger reconciliation sweep (proses trade_outcomes terlewat)."""
    n = await attribution_engine.reconciliation_sweep()
    return {"processed": n}


@app.get("/attribution/summary", response_model=dict[str, Any])
async def attribution_summary():
    """Ringkasan attribution: akurasi sinyal per regime + drift MAE."""
    from sqlalchemy import text as _text
    from shared.db.session import AsyncSessionLocal as _S
    async with _S() as db:
        by_regime = await db.execute(_text("""
            SELECT COALESCE(regime,'unknown') regime,
                   COUNT(*) n,
                   SUM(CASE WHEN signal_correct THEN 1 ELSE 0 END) correct,
                   AVG(ABS(drift_contribution)) drift_mae
            FROM attribution_results GROUP BY regime ORDER BY n DESC
        """))
        rows = [dict(r._mapping) for r in by_regime]
        totals = await db.execute(_text("""
            SELECT COUNT(*) n,
                   SUM(CASE WHEN signal_correct THEN 1 ELSE 0 END) correct,
                   AVG(ABS(drift_contribution)) drift_mae
            FROM attribution_results
        """))
        t = totals.first()
    return {
        "total": int(t.n or 0),
        "signal_accuracy": round((t.correct or 0) / t.n, 4) if t and t.n else None,
        "drift_mae": round(float(t.drift_mae), 4) if t and t.drift_mae is not None else None,
        "by_regime": [
            {
                "regime": r["regime"],
                "n": int(r["n"]),
                "accuracy": round(r["correct"] / r["n"], 4) if r["n"] else None,
                "drift_mae": round(float(r["drift_mae"]), 4) if r["drift_mae"] is not None else None,
            } for r in rows
        ],
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
