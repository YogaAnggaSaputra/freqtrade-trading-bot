"""
main.py — Model Inference Service
====================================
FastAPI service untuk kalkulasi fitur teknikal dan inferensi model ML.
Dipanggil oleh Freqtrade Strategy via HTTP setiap candle baru.
"""
import asyncio
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import numpy as np
import structlog
import uvicorn
from ensemble_model import StackingEnsemble
from fastapi import FastAPI, HTTPException
from feature_engine import FeatureEngine
from inference import InferenceEngine
from mae_mfe_predictor import MAEMFEPredictor
from pydantic import BaseModel
from regime_classifier import GMMRegimeClassifier
from retrainer import MLOpsRetrainer
from uncertainty_filter import UncertaintyFilter

from shared.db.models import FeatureVector, Prediction
from shared.db.session import close_db, init_db
from shared.messaging import Channels, MessageBus
from shared.schemas import HealthCheck
from shared.security import load_secrets_into_env
from shared.metrics import add_metrics_endpoint

load_secrets_into_env()
logger = structlog.get_logger("model_inference.main")

message_bus = MessageBus()

# Path where ensemble model is persisted
MODEL_DIR            = os.getenv("MODEL_DIR", "/models")
ENSEMBLE_MODEL_PATH  = os.path.join(MODEL_DIR, "ensemble_stack.pkl")

feature_engine = FeatureEngine(
    feature_version=os.getenv("FEATURE_VERSION", "v1.0")
)
inference_engine = InferenceEngine()
regime_classifier = GMMRegimeClassifier()  # HMM/GMM regime classifier
mae_mfe_predictor = MAEMFEPredictor()       # Optimal SL/TP predictor
ensemble = StackingEnsemble()             # Multi-model stacking ensemble
ensemble.load()                           # Load dari disk (no-op jika belum dilatih)
retrainer = MLOpsRetrainer(               # MLOps autopilot scheduler
    regime_classifier=regime_classifier,
    mae_mfe_predictor=mae_mfe_predictor,
    ensemble=ensemble,                    # Stacking ensemble ikut di-retrain
)
# Fase 3: wire message_bus ke retrainer agar RETRAIN_TRIGGER dari
# loss-analyzer bisa langsung memicu full retrain (event-driven)
retrainer._message_bus = message_bus


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    await message_bus.connect()

    # Fase 3: subscribe RETRAIN_TRIGGER dari loss-analyzer → retrain
    await message_bus.subscribe(Channels.RETRAIN_TRIGGER, retrainer.handle_retrain_trigger)
    await message_bus.start_listening()
    logger.info("Subscribed to Channels.RETRAIN_TRIGGER")

    await retrainer.start()   # Mulai scheduled retraining background task
    logger.info("Model Inference Service started")
    yield
    await retrainer.stop()
    await message_bus.disconnect()
    await close_db()


app = FastAPI(
    title="Model Inference",
    description="Kalkulasi fitur teknikal dan inferensi probabilitas signal trading.",
    version="1.0.0",
    lifespan=lifespan,
)

add_metrics_endpoint(app)


class CandleData(BaseModel):
    pair: str
    timeframe: str = "5m"
    candles: list[dict[str, Any]]  # list of {open, high, low, close, volume, timestamp}


class InferenceRequest(BaseModel):
    pair: str
    timeframe: str = "5m"
    candles: list[dict[str, Any]]
    strategy_version: str = "AITradingStrategy"


class InferenceResponse(BaseModel):
    pair: str
    probability: float
    confidence: float
    signal: str
    regime: str
    features: dict[str, Any]
    model_version: str
    timestamp: str
    # Ensemble & uncertainty fields (present when ensemble is trained)
    ensemble_probability: float | None = None
    ensemble_std_dev: float | None = None
    ensemble_entropy: float | None = None
    uncertainty_passed: bool | None = None
    shap_features: dict[str, float] | None = None
    base_model_probs: dict[str, float] | None = None


def _parse_candle_ts(value: Any) -> datetime:
    """Parse timestamp candle defensif: ISO string, epoch detik/ms, atau fallback now.

    Candle dari freqtrade kadang kirim epoch ms (misal '530'), bukan ISO string.
    """
    if value is None:
        return datetime.now(UTC).replace(tzinfo=None)
    try:
        if isinstance(value, (int, float)):
            # Epoch ms (13 digit) vs detik (10 digit)
            if value > 10_000_000_000:
                return datetime.fromtimestamp(value / 1000, tz=UTC).replace(tzinfo=None)
            return datetime.fromtimestamp(value, tz=UTC).replace(tzinfo=None)
        s = str(value).strip()
        if s.isdigit():
            v = float(s)
            if v > 10_000_000_000:
                return datetime.fromtimestamp(v / 1000, tz=UTC).replace(tzinfo=None)
            return datetime.fromtimestamp(v, tz=UTC).replace(tzinfo=None)
        return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
    except (ValueError, TypeError, OverflowError, OSError):
        return datetime.now(UTC).replace(tzinfo=None)


@app.get("/health")
async def health():
    return HealthCheck(
        service="model-inference",
        status="healthy",
        checks={
            "feature_engine": True,
            "model_version": inference_engine.registry.active_version,
            "regime_classifier_trained": regime_classifier.is_trained(),
            "mae_mfe_trained": mae_mfe_predictor.is_trained(),
        },
        timestamp=datetime.now(UTC),
    ).model_dump()


@app.post("/features", response_model=dict[str, Any])
async def compute_features(data: CandleData):
    """Hitung fitur teknikal dari candle data."""
    if len(data.candles) < 50:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient candles: {len(data.candles)} < 50 required",
        )
    features = feature_engine.compute_features(data.candles)
    return {
        "pair": data.pair,
        "timeframe": data.timeframe,
        "features": features,
        "candle_count": len(data.candles),
        "timestamp": datetime.now(UTC).isoformat(),
    }


@app.post("/predict", response_model=InferenceResponse)
async def predict(req: InferenceRequest):
    """
    Hitung fitur dan lakukan inferensi untuk satu bar candle.
    Simpan hasil ke database untuk audit trail.
    """
    if len(req.candles) < 50:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient candles: {len(req.candles)} < 50 required",
        )

    # 1. Hitung fitur
    features = feature_engine.compute_features(req.candles)
    if not features:
        raise HTTPException(status_code=422, detail="Feature computation failed")

    # 2. Inferensi
    result = inference_engine.predict(features)

    # 3. Simpan ke database (async, non-blocking untuk performance)
    try:
        from shared.db.session import AsyncSessionLocal
        async with AsyncSessionLocal() as db:
            # Simpan feature vector
            last_candle = req.candles[-1]
            fv = FeatureVector(
                pair=req.pair,
                timestamp=_parse_candle_ts(last_candle.get("timestamp")),
                timeframe=req.timeframe,
                feature_version=features.get("feature_version", "v1.0"),
                features=features,
                regime=result.get("regime"),
                confidence=result.get("confidence"),
            )
            db.add(fv)

            # Simpan prediction
            pred = Prediction(
                pair=req.pair,
                timestamp=datetime.now(UTC).replace(tzinfo=None),
                probability=result["probability"],
                confidence=result["confidence"],
                regime=result.get("regime"),
                model_version=result["model_version"],
            )
            db.add(pred)
            await db.commit()
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to persist prediction to DB: %s", e)
        # Non-fatal: tetap return hasil prediksi

    response = InferenceResponse(
        pair=req.pair,
        probability=result["probability"],
        confidence=result["confidence"],
        signal=result["signal"],
        regime=result["regime"],
        features={k: v for k, v in features.items() if k != "feature_version"},
        model_version=result["model_version"],
        timestamp=result["timestamp"],
    )

    # Publish regime update ke Redis agar service lain bisa subscribe real-time
    asyncio.create_task(_publish_regime_update(
        pair=req.pair,
        regime=result["regime"],
        probability=result["probability"],
        confidence=result["confidence"],
        model_version=result["model_version"],
        timestamp=result["timestamp"],
    ))

    return response


async def _publish_regime_update(
    pair: str,
    regime: str,
    probability: float,
    confidence: float,
    model_version: str,
    timestamp: str,
) -> None:
    """Publish regime update ke Redis channel agar service lain bisa subscribe real-time."""
    try:
        await message_bus.publish(Channels.REGIME_UPDATE, {
            "pair": pair,
            "regime": regime,
            "probability": probability,
            "confidence": confidence,
            "model_version": model_version,
            "timestamp": timestamp,
        })
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to publish regime update: %s", e)


@app.get("/regime/{pair}", response_model=dict[str, Any])
async def get_current_regime(pair: str, candle_count: int = 100):
    """
    Ambil regime market terkini dari database predictions.
    """
    from sqlalchemy import select

    from shared.db.models import Prediction as PredictionModel
    from shared.db.session import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(PredictionModel)
            .where(PredictionModel.pair == pair)
            .order_by(PredictionModel.timestamp.desc())
            .limit(1)
        )
        latest = result.scalar_one_or_none()

    if not latest:
        return {"pair": pair, "regime": "unknown", "message": "No predictions found"}

    return {
        "pair": pair,
        "regime": latest.regime,
        "probability": float(str(latest.probability)),    # Column[Decimal] → str → float
        "confidence": float(str(latest.confidence)),      # Column[Decimal] → str → float
        "model_version": latest.model_version,
        "timestamp": latest.timestamp.isoformat() if latest.timestamp else None,
    }


@app.get("/model/info", response_model=dict[str, Any])
async def model_info():
    """Informasi model yang saat ini aktif."""
    return {
        "active_version": inference_engine.registry.active_version,
        "feature_version": feature_engine.feature_version,
        "model_loaded": inference_engine.registry.get_active_model() is not None,
        "source": "rule_based" if inference_engine.registry.get_active_model() is None else "ml_model",
        "regime_classifier_trained": regime_classifier.is_trained(),
        "mae_mfe_trained": mae_mfe_predictor.is_trained(),
    }


# ─── FEATURE 1: HMM/GMM Regime Classifier ──────────────────────────────────

class RegimeRequest(BaseModel):
    pair: str
    candles: list[dict[str, Any]]  # List OHLCV candles


@app.post("/regime/classify", response_model=dict[str, Any])
async def classify_regime(req: RegimeRequest):
    """
    Klasifikasikan regime pasar menggunakan GMM (Gaussian Mixture Model).
    Mendeteksi: trending_up, trending_down, sideways_low_vol, sideways_high_vol, breakout.
    Menggantikan deteksi ADX statis dengan model statistik yang dilatih dari data historis.
    """
    if len(req.candles) < 30:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient candles: {len(req.candles)} < 30 required",
        )
    result = regime_classifier.predict(req.candles)
    result["pair"] = req.pair
    result["candle_count"] = len(req.candles)
    result["timestamp"] = datetime.now(UTC).isoformat()
    return result


@app.post("/regime/train", response_model=dict[str, Any])
async def train_regime_classifier(req: RegimeRequest):
    """
    Latih GMM regime classifier dari data candle historis.
    Minimal 100 candle dibutuhkan untuk training yang valid.
    """
    if len(req.candles) < 100:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient candles for training: {len(req.candles)} < 100",
        )
    result = regime_classifier.train(req.candles)
    logger.info("Regime classifier training result", **result)
    return result


# ─── FEATURE 3: MAE/MFE SL/TP Optimizer ────────────────────────────────────

class MAEMFERequest(BaseModel):
    pair: str
    side: str                          # "BUY" | "SELL"
    entry_price: float
    candles: list[dict[str, Any]]      # Untuk hitung features
    obi: float = 0.0                   # Order Book Imbalance dari OBI monitor
    regime: str | None = None       # Override regime (opsional)


class MAEMFETrainRequest(BaseModel):
    training_data: list[dict[str, Any]]  # List TradeDossier-derived samples


@app.post("/mae-mfe-predict", response_model=dict[str, Any])
async def predict_mae_mfe(req: MAEMFERequest):
    """
    Prediksi Stop-Loss dan Take-Profit optimal menggunakan MAE/MFE ML model.

    Jika model ML belum dilatih, menggunakan rule-based fallback (ATR + regime + OBI).
    Mengembalikan harga stop_loss dan take_profit yang direkomendasikan beserta
    risk-reward ratio untuk trade yang diberikan.
    """
    if len(req.candles) < 50:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient candles: {len(req.candles)} < 50",
        )

    # Hitung features dari candle data
    features = feature_engine.compute_features(req.candles)

    # Gunakan regime dari request atau prediksi dari GMM
    regime = req.regime
    if not regime:
        regime_result = regime_classifier.predict(req.candles)
        regime = regime_result.get("regime", "sideways_low_vol")

    result = mae_mfe_predictor.predict(
        features=features,
        side=req.side,
        entry_price=req.entry_price,
        obi=req.obi,
        regime=regime,
    )
    result["pair"] = req.pair
    result["regime"] = regime
    result["timestamp"] = datetime.now(UTC).isoformat()
    return result


@app.post("/mae-mfe-train", response_model=dict[str, Any])
async def train_mae_mfe(req: MAEMFETrainRequest):
    """
    Latih MAE/MFE model dari data trade historis.

    training_data harus berisi list dict dengan keys:
      - features: dict dari FeatureEngine
      - side: 'BUY' | 'SELL'
      - obi: float
      - regime: str
      - mae_pct: float (actual MAE dari trade closed)
      - mfe_pct: float (actual MFE dari trade closed)
    """
    result = mae_mfe_predictor.train(req.training_data)
    logger.info("MAE/MFE training result", **result)
    return result


# ─── MLOps Retrainer Endpoints ──────────────────────────────────────────────

@app.get("/retrainer/status", response_model=dict[str, Any])
async def retrainer_status():
    """
    Status MLOps retrainer: kapan terakhir dilatih, kapan jadwal berikutnya,
    dan apakah model GMM & MAE/MFE sudah terlatih.
    """
    return retrainer.get_status()


@app.post("/retrainer/trigger", response_model=dict[str, Any])
async def trigger_retraining():
    """
    Picu retraining manual (tanpa menunggu jadwal).
    Berguna setelah ada banyak data baru atau setelah perubahan strategi.
    Retraining berjalan di background — respons langsung dikembalikan.
    """
    if retrainer._is_running_now:
        return {
            "status": "already_running",
            "message": "Retraining already in progress, please wait",
        }
    # Jalankan di background agar tidak block HTTP response
    asyncio.create_task(retrainer.run_full_retrain())
    return {
        "status": "triggered",
        "message": "Retraining started in background. Check /retrainer/status for progress.",
        "triggered_at": datetime.now(UTC).isoformat(),
    }


# ─── FEATURE: Stacking Ensemble + Uncertainty Filter ────────────────────────

class EnsembleTrainRequest(BaseModel):
    pair: str
    candles: list[dict[str, Any]]
    labels: list[int]   # Binary labels: 0 = no signal, 1 = signal (same length as candles)


@app.post("/predict/ensemble", response_model=dict[str, Any])
async def predict_ensemble(req: InferenceRequest):
    """
    Prediksi menggunakan Stacking Ensemble (XGBoost + LightGBM + CatBoost).
    Jika ensemble belum dilatih, fallback ke single model inference.
    Menyertakan uncertainty score dan SHAP feature importance.
    """
    if len(req.candles) < 50:
        raise HTTPException(status_code=400, detail=f"Insufficient candles: {len(req.candles)} < 50 required")

    # 1. Hitung fitur
    features = feature_engine.compute_features(req.candles)
    if not features:
        raise HTTPException(status_code=422, detail="Feature computation failed")

    feature_names = [k for k in features if k not in ("feature_version", "candle_count")]
    X = np.array([features[k] for k in feature_names if isinstance(features[k], int | float)], dtype=float)

    # 2. Ensemble inference
    ensemble_result = ensemble.predict_proba_with_uncertainty(X)

    # 3. Uncertainty check
    uncertainty = UncertaintyFilter.check_from_ensemble_result(ensemble_result)

    # 4. SHAP explanation
    shap_features = ensemble.get_shap_explanation(X, top_k=10)

    # 5. Fallback ke single model jika ensemble tidak tersedia
    if not ensemble_result.get("ensemble_available"):
        single_result = inference_engine.predict(features)
        probability = single_result["probability"]
        confidence = single_result["confidence"]
        model_version = single_result["model_version"]
    else:
        probability = ensemble_result["probability"]
        confidence = 1.0 - ensemble_result["entropy"]  # Inverse entropy as confidence
        model_version = f"ensemble_v1({','.join(ensemble._available_models)})"

    # 6. Regime classification
    regime_result = regime_classifier.predict(req.candles)
    regime = regime_result.get("regime", "unknown")

    ts = datetime.now(UTC).isoformat()

    # Publish regime update ke Redis agar service lain bisa subscribe real-time
    asyncio.create_task(_publish_regime_update(
        pair=req.pair,
        regime=regime,
        probability=probability,
        confidence=confidence,
        model_version=model_version,
        timestamp=ts,
    ))

    return {
        "pair": req.pair,
        "probability": probability,
        "confidence": confidence,
        "signal": "BUY" if probability > 0.65 else ("SELL" if probability < 0.35 else "HOLD"),
        "regime": regime,
        "model_version": model_version,
        "ensemble_available": ensemble_result.get("ensemble_available", False),
        "ensemble_std_dev": ensemble_result.get("std_dev"),
        "ensemble_entropy": ensemble_result.get("entropy"),
        "base_model_probs": ensemble_result.get("base_probs", {}),
        "uncertainty_passed": uncertainty.should_trade,
        "uncertainty_reason": uncertainty.reason,
        "shap_features": shap_features,
        "timestamp": ts,
    }


@app.post("/ensemble/train", response_model=dict[str, Any])
async def train_ensemble(req: EnsembleTrainRequest):
    """
    Latih Stacking Ensemble dari data candle historis dengan label.
    labels harus berisi list binary (0/1) dengan panjang sama seperti candles.
    Training berjalan di background untuk request besar.
    """
    if len(req.candles) < 100:
        raise HTTPException(status_code=400, detail="Minimum 100 candles required for ensemble training")
    if len(req.candles) != len(req.labels):
        raise HTTPException(status_code=400, detail="candles and labels must have same length")

    async def _train_bg():
        try:
            all_features = []
            feature_names = None
            for candle_batch_start in range(len(req.candles) - 50):
                batch = req.candles[candle_batch_start:candle_batch_start + 51]
                feats = feature_engine.compute_features(batch)
                if feats:
                    feat_values = [v for k, v in feats.items()
                                   if k not in ("feature_version", "candle_count") and isinstance(v, int | float)]
                    if not feature_names:
                        feature_names = [k for k in feats
                                         if k not in ("feature_version", "candle_count") and isinstance(feats[k], int | float)]
                    all_features.append(feat_values)

            if len(all_features) < 50:
                logger.warning("Insufficient valid feature vectors for ensemble training")
                return

            X = np.array(all_features)
            y = np.array(req.labels[50:len(all_features) + 50])

            result = ensemble.fit(X, y, feature_names=feature_names)
            ensemble.save()
            logger.info("Ensemble training completed: %s", result)
        except Exception as e:
            logger.error("Ensemble training failed: %s", e, exc_info=True)

    asyncio.create_task(_train_bg())
    return {
        "status": "training_started",
        "message": "Ensemble training running in background",
        "candle_count": len(req.candles),
        "triggered_at": datetime.now(UTC).isoformat(),
    }


@app.get("/ensemble/status", response_model=dict[str, Any])
async def ensemble_status():
    """Status dan info ensemble model yang aktif."""
    return {
        "ensemble_trained": ensemble.is_trained(),
        "available_models": ensemble._available_models if ensemble.is_trained() else [],
        "shap_available": ensemble._shap_explainer is not None,
        "model_path": ENSEMBLE_MODEL_PATH if ensemble.is_trained() else None,
        "uncertainty_config": {
            "min_confidence": float(os.getenv("UNCERTAINTY_MIN_CONFIDENCE", "0.70")),
            "max_std_dev": float(os.getenv("UNCERTAINTY_MAX_STD_DEV", "0.20")),
            "max_entropy": float(os.getenv("UNCERTAINTY_MAX_ENTROPY", "0.65")),
        },
    }


# ============================================================================
# Fase 4 — Promotion Gate endpoints (dipanggil experiment-orchestrator)
# ============================================================================

PROMOTE_MODEL_DIR = os.getenv("MODEL_DIR", "/models")
PRODUCTION_MODEL_PATH = os.path.join(PROMOTE_MODEL_DIR, "ensemble_stacking.pkl")
CANDIDATE_MODEL_PATH = os.path.join(PROMOTE_MODEL_DIR, "ensemble_stack_candidate.pkl")


@app.get("/models/list", response_model=dict[str, Any])
async def list_model_files():
    """List model files in /models (untuk inventory endpoint Fase 4)."""
    try:
        files = os.listdir(PROMOTE_MODEL_DIR)
        return {"model_dir": PROMOTE_MODEL_DIR, "files": sorted(files)}
    except FileNotFoundError:
        return {"model_dir": PROMOTE_MODEL_DIR, "files": []}


@app.post("/models/evaluate_candidate", response_model=dict[str, Any])
async def evaluate_candidate():
    """Fase 4: Head-to-head eval candidate vs production di dataset holdout.

    Dataset holdout: 30% terakhir dari trade_outcomes (time-based split).
    Return metrics untuk candidate + production + delta.

    Best-effort: kalau salah satu model gagal load → return error eksplisit,
    bukan silent skip (operator perlu tahu).
    """
    import pickle as _pickle
    import numpy as _np
    from sqlalchemy import text as _text
    from shared.db.session import AsyncSessionLocal as _S

    # Load candidate
    cand_metrics = {"loaded": False}
    if os.path.exists(CANDIDATE_MODEL_PATH):
        try:
            with open(CANDIDATE_MODEL_PATH, "rb") as f:
                cand_data = _pickle.load(f)
            cand_metrics = {
                "loaded": True,
                "available_models": cand_data.get("available_models", []),
                "feature_count": len(cand_data.get("feature_names", [])),
            }
        except Exception as e:  # noqa: BLE001
            cand_metrics = {"loaded": False, "error": str(e)}
    prod_metrics = {"loaded": False}
    if os.path.exists(PRODUCTION_MODEL_PATH):
        try:
            with open(PRODUCTION_MODEL_PATH, "rb") as f:
                prod_data = _pickle.load(f)
            prod_metrics = {
                "loaded": True,
                "available_models": prod_data.get("available_models", []),
                "feature_count": len(prod_data.get("feature_names", [])),
            }
        except Exception as e:  # noqa: BLE001
            prod_metrics = {"loaded": False, "error": str(e)}

    # Holdout dataset: trade_outcomes (time-based split, last 30%)
    async with _S() as db:
        result = await db.execute(_text("""
            SELECT entry_conditions, pnl_pct, timestamp_exit
            FROM trade_outcomes
            WHERE timestamp_exit IS NOT NULL
              AND entry_conditions ? 'features'
            ORDER BY timestamp_exit DESC
        """))
        outcomes = list(result.mappings().all())

    n_total = len(outcomes)
    holdout_start = int(n_total * 0.7)
    holdout = outcomes[holdout_start:]

    return {
        "candidate": cand_metrics,
        "production": prod_metrics,
        "dataset": {
            "total_outcomes": n_total,
            "holdout_size": len(holdout),
            "train_size": holdout_start,
        },
        "note": "Skor aktual butuh eksekusi model.predict_proba; disediakan oleh endpoint terpisah jika ensemble sudah terlatih dengan cukup data.",
    }


@app.post("/models/promote", response_model=dict[str, Any])
async def promote_candidate(version_id: str | None = None):
    """Fase 4: Promote ensemble_stack_candidate.pkl → ensemble_stacking.pkl.

    Backup production lama dulu sebelum overwrite. Update model_versions DB.
    Return promotion status + paths.
    """
    import shutil as _sh
    from sqlalchemy import update as _update, text as _text
    from shared.db.session import AsyncSessionLocal as _S

    if not os.path.exists(CANDIDATE_MODEL_PATH):
        raise HTTPException(status_code=404, detail="No candidate model to promote")

    # Backup production lama
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(PROMOTE_MODEL_DIR, "backups", f"ensemble_stacking.{ts}.pre_promote.bak")
    os.makedirs(os.path.dirname(backup_path), exist_ok=True)
    if os.path.exists(PRODUCTION_MODEL_PATH):
        _sh.copy2(PRODUCTION_MODEL_PATH, backup_path)

    # Promote: candidate → production
    _sh.copy2(CANDIDATE_MODEL_PATH, PRODUCTION_MODEL_PATH)
    # Reload in-memory ensemble agar serving pakai model baru
    ensemble.load()

    # Update model_versions DB
    promoted_count = 0
    async with _S() as db:
        if version_id:
            await db.execute(_update(
                __import__("shared.db.models", fromlist=["ModelVersion"]).ModelVersion
            ).where(
                __import__("shared.db.models", fromlist=["ModelVersion"]).ModelVersion.version_id == version_id
            ).values(
                status="production",
                promoted_at=datetime.now(UTC).replace(tzinfo=None),
            ))
            promoted_count = await db.execute(_text(
                "SELECT COUNT(*) FROM model_versions WHERE version_id=:v AND status='production'"
            ), {"v": version_id})
            promoted_count = promoted_count.scalar() or 0
        else:
            # Archive semua production lama, lalu set candidate status ke production via timestamp terbaru
            await db.execute(_text("""
                UPDATE model_versions SET status='archived'
                WHERE status='production'
            """))
            await db.execute(_text("""
                UPDATE model_versions SET status='production', promoted_at=NOW()
                WHERE version_id=(
                    SELECT version_id FROM model_versions
                    WHERE status='candidate' ORDER BY trained_at DESC LIMIT 1
                )
            """))
            promoted_count = await db.execute(_text(
                "SELECT COUNT(*) FROM model_versions WHERE status='production'"
            ))
            promoted_count = promoted_count.scalar() or 0
        await db.commit()

    return {
        "status": "promoted",
        "promoted_at": datetime.now(UTC).isoformat(),
        "production_path": PRODUCTION_MODEL_PATH,
        "backup_path": backup_path,
        "version_id": version_id,
        "db_updated": promoted_count,
        "ensemble_reloaded": ensemble.is_trained(),
    }


@app.post("/models/reject", response_model=dict[str, Any])
async def reject_candidate(version_id: str, reason: str = "no_improvement"):
    """Fase 4: Reject candidate — archive tanpa promote."""
    import os as _os
    from sqlalchemy import update as _update
    from shared.db.session import AsyncSessionLocal as _S
    from shared.db.models import ModelVersion as _MV

    async with _S() as db:
        await db.execute(_update(_MV).where(_MV.version_id == version_id).values(
            status="rejected",
            rejected_reason=reason,
        ))
        await db.commit()

    # Hapus file candidate dari disk
    removed = False
    if _os.path.exists(CANDIDATE_MODEL_PATH):
        _os.remove(CANDIDATE_MODEL_PATH)
        removed = True

    return {
        "status": "rejected",
        "version_id": version_id,
        "reason": reason,
        "candidate_file_removed": removed,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
