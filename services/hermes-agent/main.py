"""
main.py — Hermes Improvement Agent
====================================
Entry point untuk Hermes Agent service.
Menyediakan REST API untuk memicu analisis manual dan melihat status,
serta menjalankan agent loop periodik di background.
"""
import asyncio
import os
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime
from typing import Any

import structlog
import uvicorn
from agent import HermesAgent
from fastapi import FastAPI, HTTPException

from shared.db.session import close_db, init_db
from shared.messaging import MessageBus
from shared.schemas import HealthCheck
from shared.security import load_secrets_into_env
from shared.metrics import add_metrics_endpoint

load_secrets_into_env()
logger = structlog.get_logger("hermes.main")

message_bus = MessageBus()
hermes_agent: HermesAgent | None = None
_agent_task: asyncio.Task | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global hermes_agent, _agent_task
    await init_db()
    await message_bus.connect()

    interval_hours = int(os.getenv("HERMES_INTERVAL_HOURS", "6"))
    lookback_days = int(os.getenv("HERMES_LOOKBACK_DAYS", "14"))
    min_samples = int(os.getenv("HERMES_MIN_SAMPLES", "20"))

    hermes_agent = HermesAgent(
        message_bus=message_bus,
        analysis_interval_hours=interval_hours,
        lookback_days=lookback_days,
        min_samples=min_samples,
    )
    _agent_task = asyncio.create_task(hermes_agent.start())
    logger.info("Hermes Agent started", interval_hours=interval_hours)

    yield

    if hermes_agent:
        await hermes_agent.stop()
    if _agent_task and not _agent_task.done():
        _agent_task.cancel()
        with suppress(asyncio.CancelledError):
            await _agent_task
    await message_bus.disconnect()
    await close_db()


app = FastAPI(
    title="Hermes Improvement Agent",
    description="Autonomous trading improvement agent — read-only analysis and proposal generation.",
    version="1.0.0",
    lifespan=lifespan,
)

add_metrics_endpoint(app)


@app.get("/health")
async def health():
    return HealthCheck(
        service="hermes-agent",
        status="healthy",
        checks={
            "agent_running": hermes_agent is not None and hermes_agent._running,
            "message_bus": message_bus.connected,
        },
        timestamp=datetime.now(UTC),
    ).model_dump()


@app.post("/analyze", response_model=dict[str, Any])
async def trigger_analysis():
    """
    Picu satu siklus analisis secara manual (tanpa menunggu interval).
    Berguna untuk debugging dan review setelah banyak trade.
    """
    if hermes_agent is None:
        raise HTTPException(status_code=503, detail="Hermes Agent not initialized")
    try:
        proposals = await hermes_agent.run_analysis_cycle()
        return {
            "success": True,
            "proposals_generated": len(proposals),
            "proposals": proposals,
            "timestamp": datetime.now(UTC).isoformat(),
        }
    except Exception as e:
        logger.error("Manual analysis failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/proposals", response_model=dict[str, Any])
async def list_proposals(
    status: str | None = None,
    strategy_version: str | None = None,
    limit: int = 20,
):
    """Lihat daftar proposal Hermes."""
    from sqlalchemy import select

    from shared.db.models import Proposal
    from shared.db.session import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        stmt = select(Proposal).order_by(Proposal.created_at.desc()).limit(limit)
        if status:
            stmt = stmt.where(Proposal.status == status)
        if strategy_version:
            stmt = stmt.where(Proposal.strategy_version == strategy_version)
        result = await db.execute(stmt)
        proposals = result.scalars().all()

    return {
        "proposals": [
            {
                "proposal_id": p.proposal_id,
                "strategy_version": p.strategy_version,
                "problem_type": p.problem_type,
                "status": p.status,
                "expected_effect": p.expected_effect,
                "created_at": p.created_at.isoformat() if p.created_at else None,
            }
            for p in proposals
        ],
        "count": len(proposals),
    }


@app.get("/proposals/{proposal_id}", response_model=dict[str, Any])
async def get_proposal(proposal_id: str):
    """Lihat detail satu proposal beserta evidence."""
    from sqlalchemy import select

    from shared.db.models import Proposal
    from shared.db.session import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Proposal).where(Proposal.proposal_id == proposal_id)
        )
        proposal = result.scalar_one_or_none()

    if not proposal:
        raise HTTPException(status_code=404, detail="Proposal not found")

    return {
        "proposal_id": proposal.proposal_id,
        "strategy_version": proposal.strategy_version,
        "problem_type": proposal.problem_type,
        "evidence": proposal.evidence,
        "proposed_change": proposal.proposed_change,
        "expected_effect": proposal.expected_effect,
        "validation_plan": proposal.validation_plan,
        "rollback_condition": proposal.rollback_condition,
        "status": proposal.status,
        "experiment_id": proposal.experiment_id,
        "created_at": proposal.created_at.isoformat() if proposal.created_at else None,
        "updated_at": proposal.updated_at.isoformat() if proposal.updated_at else None,
    }


@app.get("/evidence/summary", response_model=dict[str, Any])
async def get_evidence_summary(days: int = 14):
    """Lihat ringkasan evidence terkini."""
    if hermes_agent is None:
        raise HTTPException(status_code=503, detail="Hermes Agent not initialized")
    collector = hermes_agent.collector
    strategy_version = await collector.get_active_strategy_version()
    loss_summary = await collector.get_loss_summary(days=days, strategy_version=strategy_version)
    regime_performance = await collector.get_regime_performance(days=days)
    calibration = await collector.get_prediction_calibration(days=days)

    return {
        "strategy_version": strategy_version,
        "lookback_days": days,
        "loss_summary": loss_summary,
        "regime_performance": regime_performance,
        "calibration": calibration,
        "timestamp": datetime.now(UTC).isoformat(),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
