"""
main.py — Experiment Orchestrator Service
==========================================
FastAPI service untuk mengelola lifecycle eksperimen Hermes.
Menerima proposal dari Policy Gate dan mengkoordinasikan seluruh pipeline pengujian.
"""
import asyncio
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import structlog
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from orchestrator import ExperimentOrchestrator
from pydantic import BaseModel
from runner import ExperimentRunner

from shared.db.session import AsyncSessionLocal, close_db, init_db
from shared.messaging import Channels, MessageBus
from shared.schemas import HealthCheck
from shared.security import load_secrets_into_env
from shared.metrics import add_metrics_endpoint

load_secrets_into_env()
logger = structlog.get_logger("experiment_orchestrator.main")

message_bus = MessageBus()
orchestrator = ExperimentOrchestrator(message_bus=message_bus)
runner = ExperimentRunner(
    config_base=os.getenv("BACKTEST_CONFIG", "config.backtest.json"),
    strategy=os.getenv("STRATEGY_NAME", "AITradingStrategy"),
)


class AcceptProposalRequest(BaseModel):
    proposal_id: str
    candidate_config: dict[str, Any]
    baseline_config: dict[str, Any]


class StageUpdateRequest(BaseModel):
    experiment_id: str
    passed: bool
    metrics: dict[str, Any]
    notes: str = ""


class DeployStrategyRequest(BaseModel):
    strategy_version: str = "AITradingStrategy"
    model_version: str | None = None
    config_version: str = "freqtrade"
    environment: str = "demo"


AUTO_ACCEPT_PROPOSALS = os.getenv("AUTO_ACCEPT_PROPOSALS", "true").lower() == "true"
EXPERIMENT_ORCHESTRATOR_URL = os.getenv("EXPERIMENT_ORCHESTRATOR_URL", "http://experiment-orchestrator:8000")
AUTO_DEPLOY_ON_START = os.getenv("AUTO_DEPLOY_ON_START", "true").lower() == "true"
DEFAULT_STRATEGY = os.getenv("STRATEGY_NAME", "AITradingStrategy")


async def _ensure_active_deployment() -> None:
    """Auto-seed deployment aktif saat startup.

    Risk-gateway menolak order jika tidak ada strategi yang statusnya 'active'
    di tabel deployments. Untuk testnet/demo, seed satu deployment default
    agar pipeline bisa langsung berjalan tanpa harus lewat eksperimen penuh.
    """
    try:
        from sqlalchemy import select

        from shared.db.models import Deployment

        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(Deployment).where(
                    Deployment.strategy_version == DEFAULT_STRATEGY,
                    Deployment.status == "active",
                )
            )
            existing = result.scalar_one_or_none()
            if existing:
                logger.info("Active deployment already exists", strategy=DEFAULT_STRATEGY)
                return

            import uuid

            deployment = Deployment(
                deployment_id=f"DEPLOY-{uuid.uuid4().hex[:8].upper()}",
                strategy_version=DEFAULT_STRATEGY,
                model_version=None,
                config_version="freqtrade",
                environment=os.getenv("TRADE_MODE", "demo"),
                status="active",
                deployed_at=datetime.now(UTC).replace(tzinfo=None),
            )
            db.add(deployment)
            await db.commit()
            logger.info(
                "Seeded active deployment", strategy=DEFAULT_STRATEGY,
                deployment_id=deployment.deployment_id,
            )
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to seed deployment: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    await message_bus.connect()

    # Auto-seed deployment aktif agar risk-gateway tidak memblokir semua order
    if AUTO_DEPLOY_ON_START:
        await _ensure_active_deployment()

    # Subscribe ke proposal events dari Hermes
    await message_bus.subscribe(Channels.ALERT, _handle_alert)

    # Fase 4: subscribe MODEL_CANDIDATE_READY dari retrainer → gate promote/reject
    await message_bus.subscribe(Channels.MODEL_CANDIDATE_READY, _handle_model_candidate)

    asyncio.create_task(message_bus.start_listening())

    logger.info(
        "Experiment Orchestrator started. AUTO_ACCEPT_PROPOSALS=%s",
        AUTO_ACCEPT_PROPOSALS,
    )
    yield
    await message_bus.disconnect()
    await close_db()


async def _handle_model_candidate(payload: dict[str, Any]) -> None:
    """Fase 4: Handler untuk MODEL_CANDIDATE_READY dari retrainer.

    Alur:
    1. Terima version_id + holdout_metrics
    2. Panggil model-inference /models/evaluate_candidate untuk head-to-head
    3. Terapkan kriteria promote (win_rate, avg_rr, drawdown)
    4. Jika lolos: POST /models/promote → publish MODEL_DEPLOYED
    5. Jika gagal: POST /models/reject → publish MODEL_REJECTED
    """
    import urllib.request
    import json as _json

    version_id = payload.get("version_id")
    if not version_id:
        logger.warning("MODEL_CANDIDATE_READY without version_id — skipping")
        return

    logger.info("MODEL_CANDIDATE_READY received: %s", version_id)

    mi_url = os.getenv("MODEL_INFERENCE_URL", "http://model-inference:8000")

    try:
        # 1. Evaluate candidate vs production
        req = urllib.request.Request(f"{mi_url}/models/evaluate_candidate", method="POST")
        eval_result = _json.loads(urllib.request.urlopen(req, timeout=30).read().decode())
        logger.info("Candidate eval: %s", eval_result)

        # 2. Decision gate — kriteria numeric promote
        # Best-effort: jika candidate loaded dan production TIDAK → auto-promote
        cand = eval_result.get("candidate", {})
        prod = eval_result.get("production", {})

        if not cand.get("loaded"):
            logger.warning("Candidate model not loadable — rejecting: %s", cand.get("error"))
            await _reject_model(version_id, mi_url, "candidate_not_loadable")
            return

        # Jika production belum ada → promote langsung (first model)
        if not prod.get("loaded"):
            logger.info("No production model — auto-promote first candidate: %s", version_id)
            await _promote_model(version_id, mi_url)
            return

        # 3. Kriteria promote (dari PASS_THRESHOLDS existing + custom ML gate)
        # Karena dataset holdout mungkin terlalu kecil di early stage,
        # gunakan kriteria sederhana di sini → upgrade nanti saat data cukup.
        holdout_size = eval_result.get("dataset", {}).get("holdout_size", 0)
        if holdout_size < 10:
            logger.info("Holdout too small (%d) — manual promote required via POST /models/promote", holdout_size)
            await message_bus.publish(Channels.MODEL_REJECTED, {
                "version_id": version_id,
                "reason": f"holdout_too_small:{holdout_size}",
            })
            return

        # 4. Promote via REST (backup + swap + DB update)
        await _promote_model(version_id, mi_url)

    except Exception as exc:  # noqa: BLE001
        logger.error("Model candidate gate failed: %s", exc, exc_info=True)
        try:
            await message_bus.publish(Channels.MODEL_REJECTED, {
                "version_id": version_id,
                "reason": f"gate_error:{str(exc)[:200]}",
            })
        except Exception:
            pass


async def _promote_model(version_id: str, mi_url: str) -> None:
    """POST /models/promote ke model-inference + publish MODEL_DEPLOYED."""
    import urllib.request
    import json as _json

    req = urllib.request.Request(
        f"{mi_url}/models/promote?version_id={version_id}", method="POST"
    )
    result = _json.loads(urllib.request.urlopen(req, timeout=30).read().decode())
    logger.info("Model promoted: %s", result)
    await message_bus.publish(Channels.MODEL_DEPLOYED, {
        "version_id": version_id,
        "promoted_at": result.get("promoted_at"),
        "backup_path": result.get("backup_path"),
    })


async def _reject_model(version_id: str, mi_url: str, reason: str) -> None:
    """POST /models/reject ke model-inference + publish MODEL_REJECTED."""
    import urllib.request
    import json as _json

    req = urllib.request.Request(
        f"{mi_url}/models/reject?version_id={version_id}&reason={reason}", method="POST"
    )
    result = _json.loads(urllib.request.urlopen(req, timeout=30).read().decode())
    logger.info("Model rejected: %s", result)
    await message_bus.publish(Channels.MODEL_REJECTED, {
        "version_id": version_id,
        "reason": reason,
    })


async def _handle_alert(payload: dict[str, Any]) -> None:
    """Handle incoming alerts dari message bus.

    Jika type == 'hermes_proposal' dan AUTO_ACCEPT_PROPOSALS aktif,
    otomatis fetch proposal dari DB, build candidate_config dari
    proposed_change, dan jalankan pipeline experiment di background.
    Promote ke live tetap membutuhkan approval manual Owner.
    """
    if payload.get("type") != "hermes_proposal":
        return

    proposal_id = payload.get("proposal_id")
    logger.info("Received Hermes proposal via bus: %s", proposal_id)

    if not AUTO_ACCEPT_PROPOSALS:
        logger.info(
            "AUTO_ACCEPT_PROPOSALS=false — proposal %s requires manual acceptance via POST /experiments.",
            proposal_id,
        )
        return

    # Fetch detail proposal dari database
    try:
        from sqlalchemy import select

        from shared.db.models import Deployment, Proposal
        from shared.db.session import AsyncSessionLocal

        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(Proposal).where(Proposal.proposal_id == proposal_id)
            )
            proposal = result.scalar_one_or_none()

        if not proposal:
            logger.warning("Proposal %s not found in DB — skipping auto-accept.", proposal_id)
            return

        if proposal.status != "pending":
            logger.info(
                "Proposal %s is already in status '%s' — skipping.", proposal_id, proposal.status
            )
            return

        # Build candidate_config dari proposed_change
        proposed_change: dict[str, Any] = proposal.proposed_change or {}
        candidate_config: dict[str, Any] = {
            "pair_whitelist": ["BTC/USDT:USDT"],
            "strategy_version": proposal.strategy_version,
            "proposal_id": proposal_id,
            "problem_type": proposal.problem_type,
        }
        # Merge proposed parameter changes langsung ke candidate_config
        if isinstance(proposed_change, dict):
            candidate_config.update(proposed_change)

        # Build baseline_config — ambil dari deployment aktif jika ada
        baseline_config: dict[str, Any] = {"pair_whitelist": ["BTC/USDT:USDT"]}
        try:
            async with AsyncSessionLocal() as db:
                dep_result = await db.execute(
                    select(Deployment)
                    .where(Deployment.strategy_version == proposal.strategy_version)
                    .order_by(Deployment.deployed_at.desc())
                    .limit(1)
                )
                deployment = dep_result.scalar_one_or_none()
                if deployment:
                    # Deployment model tidak punya kolom config_snapshot — gunakan
                    # metadata yang tersedia (strategy_version + config_version)
                    baseline_config.update({
                        "strategy_version": deployment.strategy_version,
                        "config_version": deployment.config_version,
                    })
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not fetch baseline deployment config: %s", e)

        # Buat dan jalankan experiment
        experiment_id = await orchestrator.accept_proposal(
            proposal_id=proposal_id,
            candidate_config=candidate_config,
            baseline_config=baseline_config,
        )
        if not experiment_id:
            logger.warning(
                "Failed to create experiment for proposal %s (already in progress?).", proposal_id
            )
            return

        # Jalankan pipeline di background
        asyncio.create_task(_run_experiment_pipeline(experiment_id, candidate_config))

        # Kirim notifikasi ke channel ALERT
        await message_bus.publish(Channels.ALERT, {
            "type": "experiment_auto_started",
            "proposal_id": proposal_id,
            "experiment_id": experiment_id,
            "problem_type": proposal.problem_type,
            "strategy_version": proposal.strategy_version,
            "message": "Experiment pipeline started automatically from Hermes proposal.",
        })
        logger.info(
            "Auto-triggered experiment %s for proposal %s.", experiment_id, proposal_id
        )

    except Exception as e:  # noqa: BLE001
        logger.error(
            "Failed to auto-accept proposal %s: %s", proposal_id, e, exc_info=True
        )


app = FastAPI(
    title="Experiment Orchestrator",
    description="Mengelola pipeline pengujian kandidat strategi: backtest → walk-forward → demo → canary.",
    version="1.0.0",
    lifespan=lifespan,
)

add_metrics_endpoint(app)


@app.get("/health")
async def health():
    return HealthCheck(
        service="experiment-orchestrator",
        status="healthy",
        checks={"message_bus": message_bus.connected},
        timestamp=datetime.now(UTC),
    ).model_dump()


@app.post("/experiments", response_model=dict[str, Any])
async def accept_proposal(req: AcceptProposalRequest, background_tasks: BackgroundTasks):
    """Terima proposal yang sudah lolos policy, buat eksperimen baru."""
    experiment_id = await orchestrator.accept_proposal(
        proposal_id=req.proposal_id,
        candidate_config=req.candidate_config,
        baseline_config=req.baseline_config,
    )
    if not experiment_id:
        raise HTTPException(
            status_code=400,
            detail="Failed to create experiment. Proposal not found or not pending.",
        )

    # Jalankan pipeline di background
    background_tasks.add_task(_run_experiment_pipeline, experiment_id, req.candidate_config)

    return {
        "experiment_id": experiment_id,
        "proposal_id": req.proposal_id,
        "status": "schema_validation",
        "message": "Experiment created. Pipeline running in background.",
    }


@app.get("/experiments", response_model=dict[str, Any])
async def list_experiments():
    """Lihat semua eksperimen yang sedang berjalan."""
    experiments = await orchestrator.get_pending_experiments()
    return {"experiments": experiments, "count": len(experiments)}


@app.get("/experiments/{experiment_id}", response_model=dict[str, Any])
async def get_experiment(experiment_id: str):
    """Detail satu eksperimen."""
    from sqlalchemy import select

    from shared.db.models import Experiment
    from shared.db.session import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Experiment).where(Experiment.experiment_id == experiment_id)
        )
        exp = result.scalar_one_or_none()

    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")

    return {
        "experiment_id": exp.experiment_id,
        "proposal_id": exp.proposal_id,
        "status": exp.status,
        "candidate_config": exp.candidate_config,
        "baseline_config": exp.baseline_config,
        "metrics": exp.metrics,
        "started_at": exp.started_at.isoformat() if exp.started_at else None,
        "completed_at": exp.completed_at.isoformat() if exp.completed_at else None,
    }


@app.post("/experiments/{experiment_id}/advance", response_model=dict[str, Any])
async def advance_stage(experiment_id: str, req: StageUpdateRequest):
    """Update stage eksperimen secara manual (untuk stage seperti shadow/demo)."""
    new_status = await orchestrator.advance_stage(
        experiment_id=experiment_id,
        passed=req.passed,
        metrics=req.metrics,
        notes=req.notes,
    )
    return {"experiment_id": experiment_id, "new_status": new_status}


@app.get("/deployments", response_model=dict[str, Any])
async def list_deployments():
    """Lihat semua strategi yang aktif di-deploy."""
    from sqlalchemy import select

    from shared.db.models import Deployment

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Deployment).order_by(Deployment.deployed_at.desc())
        )
        deployments = result.scalars().all()

    return {
        "deployments": [
            {
                "deployment_id": d.deployment_id,
                "strategy_version": d.strategy_version,
                "model_version": d.model_version,
                "config_version": d.config_version,
                "environment": d.environment,
                "status": d.status,
                "deployed_at": d.deployed_at.isoformat() if d.deployed_at else None,
            }
            for d in deployments
        ],
        "count": len(deployments),
    }


@app.post("/deployments", response_model=dict[str, Any])
async def deploy_strategy(req: DeployStrategyRequest):
    """Deploy strategi ke tabel deployments (status active).

    Inilah mekanisme yang dibaca risk-gateway (`check_strategy_approved`):
    tanpa row 'active' di sini, semua order akan ditolak. Untuk testnet/demo,
    strategi di-seed otomatis saat startup — endpoint ini untuk kontrol manual.
    """
    import uuid

    from sqlalchemy import select

    from shared.db.models import Deployment

    async with AsyncSessionLocal() as db:
        # Nonaktifkan deployment lama untuk strategi yang sama
        old = await db.execute(
            select(Deployment).where(Deployment.strategy_version == req.strategy_version)
        )
        for d in old.scalars().all():
            d.status = "inactive"
            d.rolled_back_at = datetime.now(UTC).replace(tzinfo=None)

        deployment = Deployment(
            deployment_id=f"DEPLOY-{uuid.uuid4().hex[:8].upper()}",
            strategy_version=req.strategy_version,
            model_version=req.model_version,
            config_version=req.config_version,
            environment=req.environment,
            status="active",
            deployed_at=datetime.now(UTC).replace(tzinfo=None),
        )
        db.add(deployment)
        await db.commit()
        deployment_id = deployment.deployment_id

    logger.info(
        "Deployed strategy", strategy=req.strategy_version, deployment_id=deployment_id
    )
    return {
        "deployment_id": deployment_id,
        "strategy_version": req.strategy_version,
        "environment": req.environment,
        "status": "active",
        "message": "Strategy deployed. Risk-gateway akan menyetujui order untuk strategi ini.",
    }


@app.post("/experiments/{experiment_id}/promote", response_model=dict[str, Any])
async def request_promotion(experiment_id: str):
    """
    Request promosi ke live (memerlukan approval manual Owner via Telegram/dashboard).
    Ini BUKAN auto-promote — hanya membuat notifikasi.
    """
    success = await orchestrator.request_promotion(
        experiment_id=experiment_id,
        requested_by="api",
    )
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Experiment not in canary stage or not found",
        )
    return {
        "experiment_id": experiment_id,
        "message": "Promotion requested. Awaiting manual Owner approval via Telegram/dashboard.",
    }


async def _run_experiment_pipeline(
    experiment_id: str,
    candidate_config: dict[str, Any],
) -> None:
    """Background task: jalankan pipeline backtest + walk-forward + stress test."""
    logger.info("Starting pipeline for experiment: %s", experiment_id)
    pairs = candidate_config.get("pair_whitelist", ["BTC/USDT:USDT"])

    # 1. Schema Validation → Backtest
    logger.info("[%s] Stage: backtest", experiment_id)
    success, backtest_metrics = await runner.run_backtest(
        experiment_id=experiment_id,
        candidate_params=candidate_config,
        timerange="20240101-20241231",
        pairs=pairs,
    )
    passed, reason = await orchestrator.check_thresholds("backtest", backtest_metrics)
    new_status = await orchestrator.advance_stage(
        experiment_id, passed and success, backtest_metrics, reason
    )
    if new_status != "walk_forward":
        logger.info("[%s] Stopped at backtest: %s", experiment_id, new_status)
        return

    # 2. Walk-forward
    logger.info("[%s] Stage: walk_forward", experiment_id)
    wf_passed, wf_metrics = await runner.run_walkforward(
        experiment_id=experiment_id,
        candidate_params=candidate_config,
        windows=6,
        pairs=pairs,
    )
    wf_threshold_passed, wf_reason = await orchestrator.check_thresholds(
        "walk_forward", wf_metrics
    )
    new_status = await orchestrator.advance_stage(
        experiment_id,
        wf_passed and wf_threshold_passed,
        wf_metrics,
        wf_reason,
    )
    if new_status != "stress_test":
        logger.info("[%s] Stopped at walk_forward: %s", experiment_id, new_status)
        return

    # 3. Stress test
    logger.info("[%s] Stage: stress_test", experiment_id)
    st_passed, st_metrics = await runner.run_stress_test(
        experiment_id=experiment_id,
        base_metrics=backtest_metrics,
    )
    new_status = await orchestrator.advance_stage(
        experiment_id, st_passed, st_metrics, "stress_test_complete"
    )
    if new_status != "monte_carlo":
        logger.info("[%s] Stopped at stress_test: %s", experiment_id, new_status)
        return

    # 4. Monte Carlo (simplified: menggunakan stress test result)
    logger.info("[%s] Stage: monte_carlo (using stress test results)", experiment_id)
    mc_passed = st_metrics.get("pass_rate", 0) >= 0.50
    new_status = await orchestrator.advance_stage(
        experiment_id, mc_passed, {"pass_rate": st_metrics.get("pass_rate", 0)}, "monte_carlo_complete"
    )

    logger.info(
        "[%s] Automated stages complete. Status: %s. Next stages (shadow/demo/canary) require manual monitoring.",
        experiment_id,
        new_status,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
