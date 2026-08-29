"""
retrainer.py
=============
MLOps Autopilot Scheduler — Automated Periodic Model Retraining.

Menjaga model AI tetap fresh tanpa intervensi manual:
  1. GMM Regime Classifier → dilatih ulang dari MarketCandle 5m (90 hari terakhir)
  2. MAE/MFE SL/TP Predictor → dilatih ulang dari TradeDossier closed (180 hari terakhir)

Safety Guarantees:
  - Model lama selalu di-backup sebelum diganti
  - Minimal sample threshold sebelum melatih (hindari overfit dari data sedikit)
  - Jika training gagal, model lama tetap dipakai (fail-safe)
  - Pertama kali berjalan: tunggu 30 menit setelah startup agar data DB siap

Konfigurasi (via env vars):
  RETRAIN_INTERVAL_DAYS=7       : Interval training otomatis (hari)
  MIN_CANDLES_FOR_GMM=500       : Minimum candle untuk train GMM
  MIN_TRADES_FOR_MAE_MFE=50     : Minimum closed trades untuk train MAE/MFE
  GMM_CANDLE_LOOKBACK_DAYS=90   : Lookback data candle untuk GMM
  MAE_MFE_LOOKBACK_DAYS=180     : Lookback closed trades untuk MAE/MFE
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
from datetime import UTC, datetime, timedelta
from typing import Any

logger = logging.getLogger("model_inference.retrainer")

MODEL_DIR = os.getenv("MODEL_DIR", "/models")
RETRAIN_INTERVAL_DAYS = int(os.getenv("RETRAIN_INTERVAL_DAYS", "7"))
MIN_CANDLES_FOR_GMM = int(os.getenv("MIN_CANDLES_FOR_GMM", "500"))
MIN_TRADES_FOR_MAE_MFE = int(os.getenv("MIN_TRADES_FOR_MAE_MFE", "50"))
GMM_CANDLE_LOOKBACK_DAYS = int(os.getenv("GMM_CANDLE_LOOKBACK_DAYS", "90"))
MAE_MFE_LOOKBACK_DAYS = int(os.getenv("MAE_MFE_LOOKBACK_DAYS", "180"))
STARTUP_DELAY_MINUTES = 30  # Tunggu sebelum retrain pertama kali
AUTO_RETRAIN_ENABLED = os.getenv("AUTO_RETRAIN_ENABLED", "true").lower() == "true"


class MLOpsRetrainer:
    """
    Scheduler otomatis untuk melatih ulang model AI secara periodik.
    Dijalankan sebagai background asyncio task di dalam model-inference service.
    """

    def __init__(self, regime_classifier: Any, mae_mfe_predictor: Any, ensemble: Any = None):
        self.regime_classifier = regime_classifier
        self.mae_mfe_predictor = mae_mfe_predictor
        self.ensemble = ensemble
        self._running = False
        self._last_retrain: datetime | None = None
        self._last_result: dict[str, Any] = {}
        self._is_running_now = False

    async def start(self) -> None:
        """Mulai background retraining loop."""
        if not AUTO_RETRAIN_ENABLED:
            logger.info("MLOps Retrainer disabled by AUTO_RETRAIN_ENABLED=false")
            return
        self._running = True
        asyncio.create_task(self._retrain_loop())
        logger.info(
            "MLOps Retrainer started. Interval: %d days. First run in %d minutes.",
            RETRAIN_INTERVAL_DAYS, STARTUP_DELAY_MINUTES,
        )

    async def stop(self) -> None:
        self._running = False

    async def _retrain_loop(self) -> None:
        """Background loop: tunggu startup delay, lalu retrain setiap N hari."""
        # Tunggu setelah startup sebelum retrain pertama agar DB siap
        await asyncio.sleep(STARTUP_DELAY_MINUTES * 60)

        while self._running:
            try:
                logger.info("Scheduled model retraining started...")
                result = await self.run_full_retrain()
                self._last_retrain = datetime.now(UTC)
                self._last_result = result
                logger.info(
                    "Scheduled retraining complete. GMM: %s | MAE/MFE: %s",
                    result.get("gmm", {}).get("status", "?"),
                    result.get("mae_mfe", {}).get("status", "?"),
                )
            except Exception as e:  # noqa: BLE001
                logger.error("Scheduled retraining failed: %s", e, exc_info=True)

            # Tunggu interval berikutnya
            await asyncio.sleep(RETRAIN_INTERVAL_DAYS * 86400)

    async def run_full_retrain(self) -> dict[str, Any]:
        """
        Jalankan full retraining: GMM Regime Classifier + MAE/MFE Predictor + Ensemble.
        Dipanggil otomatis oleh scheduler atau secara manual via /retrainer/trigger.
        """
        if self._is_running_now:
            return {"status": "already_running", "message": "Retraining already in progress"}

        self._is_running_now = True
        result: dict[str, Any] = {
            "started_at": datetime.now(UTC).isoformat(),
            "gmm": {},
            "mae_mfe": {},
            "ensemble": {},
        }

        try:
            # 1. GMM Regime Classifier
            try:
                result["gmm"] = await self._retrain_gmm()
            except Exception as e:  # noqa: BLE001
                result["gmm"] = {"status": "error", "error": str(e)}
                logger.error("GMM retrain error: %s", e)

            # 2. MAE/MFE Predictor
            try:
                result["mae_mfe"] = await self._retrain_mae_mfe()
            except Exception as e:  # noqa: BLE001
                result["mae_mfe"] = {"status": "error", "error": str(e)}
                logger.error("MAE/MFE retrain error: %s", e)

            # 3. Stacking Ensemble (label = profit > 0)
            try:
                result["ensemble"] = await self._retrain_ensemble()
            except Exception as e:  # noqa: BLE001
                result["ensemble"] = {"status": "error", "error": str(e)}
                logger.error("Ensemble retrain error: %s", e)
        finally:
            self._is_running_now = False

        result["completed_at"] = datetime.now(UTC).isoformat()
        return result

    # ─── GMM Retraining ────────────────────────────────────────────────────────

    async def _retrain_gmm(self) -> dict[str, Any]:
        """Latih ulang GMM Regime Classifier dari MarketCandle 5m terbaru."""
        from sqlalchemy import select

        from shared.db.models import MarketCandle
        from shared.db.session import AsyncSessionLocal

        cutoff = datetime.now(UTC).replace(tzinfo=None) - timedelta(
            days=GMM_CANDLE_LOOKBACK_DAYS
        )

        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(MarketCandle)
                .where(
                    MarketCandle.timeframe == "5m",
                    MarketCandle.timestamp >= cutoff,
                )
                .order_by(MarketCandle.timestamp.asc())
                .limit(10000)
            )
            db_candles = result.scalars().all()

        n_candles = len(db_candles)
        if n_candles < MIN_CANDLES_FOR_GMM:
            return {
                "status": "skipped",
                "reason": f"Insufficient 5m candles: {n_candles} < {MIN_CANDLES_FOR_GMM}",
                "candles_available": n_candles,
            }

        candles = [
            {
                "open": float(c.open),
                "high": float(c.high),
                "low": float(c.low),
                "close": float(c.close),
                "volume": float(c.volume),
            }
            for c in db_candles
        ]

        # Backup model lama sebelum diganti
        self._backup_model("gmm_regime.pkl")

        train_result = self.regime_classifier.train(candles)
        train_result["candles_used"] = n_candles
        train_result["lookback_days"] = GMM_CANDLE_LOOKBACK_DAYS
        return train_result

    # ─── MAE/MFE Retraining ────────────────────────────────────────────────────

    async def _retrain_mae_mfe(self) -> dict[str, Any]:
        """Latih ulang MAE/MFE dari TradeDossier yang sudah closed."""
        from sqlalchemy import select

        from shared.db.models import TradeDossier as TradeDossierDB
        from shared.db.session import AsyncSessionLocal

        cutoff = datetime.now(UTC).replace(tzinfo=None) - timedelta(
            days=MAE_MFE_LOOKBACK_DAYS
        )

        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(TradeDossierDB)
                .where(
                    TradeDossierDB.closed_at.isnot(None),
                    TradeDossierDB.closed_at >= cutoff,
                )
                .order_by(TradeDossierDB.closed_at.desc())
                .limit(2000)
            )
            dossiers = result.scalars().all()

        n_dossiers = len(dossiers)
        if n_dossiers < MIN_TRADES_FOR_MAE_MFE:
            # Fallback: pakai trade_outcomes (fitur backfill + sl_pct/predicted_rr)
            fb = await self._mae_mfe_from_outcomes()
            if fb is not None:
                return fb
            return {
                "status": "skipped",
                "reason": f"Insufficient closed trades: {n_dossiers} < {MIN_TRADES_FOR_MAE_MFE}",
                "trades_available": n_dossiers,
            }

        # Konversi TradeDossier → training samples
        training_data: list[dict[str, Any]] = []
        for d in dossiers:
            sample = await self._dossier_to_training_sample(d)
            if sample:
                training_data.append(sample)

        if not training_data:
            return {
                "status": "skipped",
                "reason": "No valid training samples extracted from dossiers",
                "dossiers_processed": n_dossiers,
            }

        # Backup models lama
        for fname in ("mae_predictor.pkl", "mfe_predictor.pkl", "mae_mfe_scaler.pkl"):
            self._backup_model(fname)

        train_result = self.mae_mfe_predictor.train(training_data)
        train_result["dossiers_processed"] = n_dossiers
        train_result["lookback_days"] = MAE_MFE_LOOKBACK_DAYS
        return train_result

    async def _mae_mfe_from_outcomes(self) -> dict[str, Any] | None:
        """Fallback MAE/MFE training dari trade_outcomes (fitur backfill +
        excursion nyata mae_pct/mfe_pct yang di-backfill dari SQLite).

        Return None kalau data tak memadai → caller pakai path 'skipped' biasa.
        """
        from sqlalchemy import text as _text

        from shared.db.session import AsyncSessionLocal

        async with AsyncSessionLocal() as db:
            rows = (await db.execute(_text("""
                SELECT entry_conditions, pnl_pct
                FROM trade_outcomes
                WHERE entry_conditions ? 'features'
                  AND (entry_conditions->'features'->>'mae_pct') IS NOT NULL
                  AND (entry_conditions->'features'->>'mfe_pct') IS NOT NULL
            """))).mappings().all()

        training_data: list[dict[str, Any]] = []
        for r in rows:
            ec = r["entry_conditions"] or {}
            feats = ec.get("features", {}) or {}
            try:
                mae = float(feats.get("mae_pct"))
                mfe = float(feats.get("mfe_pct"))
            except (TypeError, ValueError):
                continue
            if not (0.0 <= mae <= 0.30) or not (0.0 <= mfe <= 0.50):
                continue
            training_data.append({
                "features": {k: v for k, v in feats.items()
                             if k not in ("mae_pct", "mfe_pct")},
                "side": "SELL" if str(ec.get("side", "")).lower() == "short" else "BUY",
                "obi": 0.0,
                "regime": ec.get("regime") or "sideways_low_vol",
                "mae_pct": mae,
                "mfe_pct": mfe,
            })

        if len(training_data) < MIN_TRADES_FOR_MAE_MFE:
            return None

        for fname in ("mae_predictor.pkl", "mfe_predictor.pkl", "mae_mfe_scaler.pkl"):
            self._backup_model(fname)
        result = self.mae_mfe_predictor.train(training_data)
        result["source"] = "trade_outcomes_fallback"
        result["samples_available"] = len(training_data)
        return result

    async def _dossier_to_training_sample(self, dossier: Any) -> dict[str, Any] | None:
        """
        Konversi satu TradeDossier closed ke format training MAE/MFE.

        MAE = jarak stop_loss dari entry (proxy untuk Maximum Adverse Excursion)
        MFE = jarak take_profit dari entry (proxy untuk Maximum Favorable Excursion)

        Jika feature_snapshot kosong (trade lama / webhook tanpa fitur), hitung
        ulang dari market_candles sebelum entry sehingga trade tetap bisa dipakai.
        """
        try:
            entry_data: dict = dossier.entry or {}
            sl_tp_data: dict = dossier.sl_tp or {}
            feature_snapshot: dict = dossier.feature_snapshot or {}
            regime = getattr(dossier, "market_regime", None) or "sideways_low_vol"

            entry_price = float(
                entry_data.get("entry_price") or entry_data.get("price") or 0
            )
            stop_loss = float(sl_tp_data.get("stop_loss") or 0)
            take_profit = float(sl_tp_data.get("take_profit") or 0)

            if entry_price <= 0 or stop_loss <= 0 or take_profit <= 0:
                return None

            # Inferensi side dari posisi SL relatif ke entry
            side = "BUY" if stop_loss < entry_price else "SELL"

            mae_pct = abs(stop_loss - entry_price) / entry_price
            mfe_pct = abs(take_profit - entry_price) / entry_price

            # Filter nilai tidak wajar
            if not (0.001 <= mae_pct <= 0.15) or not (0.001 <= mfe_pct <= 0.30):
                return None

            # Fallback: hitung ulang fitur dari candles jika snapshot kosong
            features = feature_snapshot or await self._compute_features_for_dossier(dossier)

            return {
                "features": features,
                "side": side,
                "obi": float(features.get("obi", 0.0)),
                "regime": regime,
                "mae_pct": mae_pct,
                "mfe_pct": mfe_pct,
            }
        except (TypeError, ValueError, AttributeError, KeyError):
            return None

    async def _compute_features_for_dossier(self, dossier: Any) -> dict[str, Any]:
        """Hitung feature snapshot dari market_candles sebelum entry trade."""
        try:
            from sqlalchemy import select

            from shared.db.models import MarketCandle
            from shared.db.session import AsyncSessionLocal

            entry_data: dict = dossier.entry or {}
            pair = entry_data.get("pair", "")
            open_date = entry_data.get("open_date", "")
            if not pair:
                return {}

            # Ambil 100 candle sebelum entry (pakai created_at sebagai fallback)
            reference_ts = None
            if open_date:
                try:
                    reference_ts = datetime.fromisoformat(str(open_date).replace("Z", "+00:00")).replace(tzinfo=None)
                except ValueError:
                    reference_ts = None
            if reference_ts is None:
                reference_ts = dossier.created_at or datetime.now(UTC).replace(tzinfo=None)

            async with AsyncSessionLocal() as db:
                result = await db.execute(
                    select(MarketCandle)
                    .where(
                        MarketCandle.pair == pair,
                        MarketCandle.timeframe == "5m",
                        MarketCandle.timestamp <= reference_ts,
                    )
                    .order_by(MarketCandle.timestamp.desc())
                    .limit(100)
                )
                db_candles = list(reversed(result.scalars().all()))

            if len(db_candles) < 50:
                return {}

            candles = [
                {
                    "open": float(c.open),
                    "high": float(c.high),
                    "low": float(c.low),
                    "close": float(c.close),
                    "volume": float(c.volume),
                    "timestamp": c.timestamp.isoformat() if c.timestamp else "",
                }
                for c in db_candles
            ]

            from feature_engine import FeatureEngine
            engine = FeatureEngine(feature_version="v1.0")
            features = engine.compute_features(candles)
            return features if features else {}
        except Exception as e:  # noqa: BLE001
            logger.warning("Feature recompute failed for dossier: %s", e)
            return {}

    # ─── Ensemble Retraining ───────────────────────────────────────────────────

    async def _retrain_ensemble(self) -> dict[str, Any]:
        """Latih Stacking Ensemble dari trade historis (label = profit > 0).

        Fase 3: dataset dari trade_outcomes (feedback loop Fase 1) sebagai
        fallback ketika trade_dossiers kosong (0 row). trade_outcomes menyimpan
        entry_conditions (JSONB fitur + regime) dan pnl_pct untuk label.
        """
        from sqlalchemy import select, text as _text

        from shared.db.session import AsyncSessionLocal

        # ── 1. Coba dataset dari trade_dossiers (existing) ──
        dossiers = await self._load_dossiers()
        if dossiers:
            return await self._train_from_dossiers(dossiers)

        # ── 2. Fallback: dataset dari trade_outcomes (Fase 1) ──
        logger.info("No TradeDossiers found, falling back to trade_outcomes...")
        async with AsyncSessionLocal() as db:
            result = await db.execute(_text("""
                SELECT trade_id, pair, entry_conditions, pnl_pct,
                       regime_at_entry, timestamp_exit
                FROM trade_outcomes
                WHERE timestamp_exit IS NOT NULL
                ORDER BY timestamp_exit DESC
                LIMIT 2000
            """))
            outcomes = result.mappings().all()

        if len(outcomes) < 1:
            return {
                "status": "skipped",
                "reason": "No data: both trade_dossiers and trade_outcomes are empty",
            }

        X, y, feature_names = [], [], None

        for row in outcomes:
            try:
                ec = row["entry_conditions"] or {}
                features = ec.get("features", {})
                if not features or len(features) < 5:
                    continue

                numeric = {
                    k: v for k, v in features.items()
                    if k not in ("feature_version", "candle_count") and isinstance(v, (int, float))
                }
                if len(numeric) < 5:
                    continue

                if feature_names is None:
                    feature_names = list(numeric.keys())
                X.append([numeric[k] for k in feature_names])
                y.append(1 if (row["pnl_pct"] or 0) > 0 else 0)
            except (TypeError, ValueError, AttributeError, KeyError):
                continue

        if len(X) < 10:
            return {
                "status": "skipped",
                "reason": f"Insufficient valid samples from trade_outcomes: {len(X)} < 10",
                "outcomes_processed": len(outcomes),
            }

        import numpy as np
        X_arr = np.array(X, dtype=float)
        y_arr = np.array(y, dtype=int)

        if len(set(y_arr)) < 2:
            return {
                "status": "skipped",
                "reason": "Need both winning and losing trades for ensemble training",
                "valid_samples": len(X_arr),
            }

        logger.info("Training ensemble from %d trade_outcomes samples", len(X_arr))

        return await self._fit_and_save_ensemble(X_arr, y_arr, feature_names, len(X_arr))

    async def _load_dossiers(self) -> list[Any]:
        """Load closed TradeDossiers untuk ensemble training (existing path)."""
        from sqlalchemy import select

        from shared.db.models import TradeDossier as TradeDossierDB
        from shared.db.session import AsyncSessionLocal

        cutoff = datetime.now(UTC).replace(tzinfo=None) - timedelta(
            days=MAE_MFE_LOOKBACK_DAYS
        )

        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(TradeDossierDB)
                .where(
                    TradeDossierDB.closed_at.isnot(None),
                    TradeDossierDB.closed_at >= cutoff,
                )
                .order_by(TradeDossierDB.closed_at.desc())
                .limit(2000)
            )
            return list(result.scalars().all())

    async def _train_from_dossiers(self, dossiers: list[Any]) -> dict[str, Any]:
        """Original ensemble training dari trade_dossiers (refactored dari versi lama)."""
        if len(dossiers) < MIN_TRADES_FOR_MAE_MFE:
            return {
                "status": "skipped",
                "reason": f"Insufficient closed trades: {len(dossiers)} < {MIN_TRADES_FOR_MAE_MFE}",
                "trades_available": len(dossiers),
            }

        X, y, feature_names = [], [], None

        for d in dossiers:
            try:
                features = (d.feature_snapshot or {}) or await self._compute_features_for_dossier(d)
                if not features:
                    continue

                numeric = {
                    k: v for k, v in features.items()
                    if k not in ("feature_version", "candle_count") and isinstance(v, (int, float))
                }
                if len(numeric) < 5:
                    continue

                if feature_names is None:
                    feature_names = list(numeric.keys())
                X.append([numeric[k] for k in feature_names])
                y.append(1 if (d.realized_pnl or 0) > 0 else 0)
            except Exception as e:  # noqa: BLE001
                logger.debug("Ensemble sample skip: %s", e)

        if len(X) < 50:
            return {
                "status": "skipped",
                "reason": f"Insufficient valid ensemble samples: {len(X)} < 50",
                "dossiers_processed": len(dossiers),
            }

        import numpy as np
        X_arr = np.array(X, dtype=float)
        y_arr = np.array(y, dtype=int)

        if len(set(y_arr)) < 2:
            return {
                "status": "skipped",
                "reason": "Need both winning and losing trades for ensemble training",
                "valid_samples": len(X_arr),
            }

        logger.info("Training ensemble from %d trade_dossiers samples", len(X_arr))

        return await self._fit_and_save_ensemble(X_arr, y_arr, feature_names, len(X_arr))

    async def _fit_and_save_ensemble(self, X_arr, y_arr, feature_names, n_samples) -> dict[str, Any]:
        """Fit ensemble, save candidate, record version, publish candidate_ready."""
        self._backup_model("ensemble_stack.pkl")

        if self.ensemble is None:
            from ensemble_model import StackingEnsemble
            self.ensemble = StackingEnsemble()

        train_result = self.ensemble.fit(X_arr, y_arr, feature_names=feature_names)

        # Fase 3: simpan candidate, BUKAN overwrite production
        candidate_path = os.path.join(MODEL_DIR, "ensemble_stack_candidate.pkl")
        self.ensemble.save(path=candidate_path)
        train_result["candidate_path"] = candidate_path
        train_result["valid_samples"] = int(n_samples)

        # Record ke model_versions + retrain_jobs
        version_id, job_id = await self._record_candidate(train_result, int(n_samples))
        train_result["version_id"] = version_id
        train_result["job_id"] = job_id

        # Publish MODEL_CANDIDATE_READY
        await self._publish_candidate_ready(version_id, train_result)
        return train_result

    # ─── Utilities ─────────────────────────────────────────────────────────────

    async def _record_candidate(self, train_result: dict[str, Any], dataset_size: int) -> tuple[str, str]:
        """Tulis ke model_versions + retrain_jobs untuk audit trail.

        Return (version_id, job_id).
        """
        from sqlalchemy import insert as _insert
        from sqlalchemy import text as _text

        from shared.db.models import ModelVersion, RetrainJob
        from shared.db.session import AsyncSessionLocal

        ts = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        version_id = f"v{ts}"
        job_id = f"job-{ts}-{os.getpid()}"

        holdout = {
            "n_samples": int(dataset_size),
            "train_metrics": {
                k: v for k, v in (train_result or {}).items()
                if k in ("cv_score", "train_accuracy", "stack_acc", "xgb_score", "lgbm_score")
            },
            "trained_at": datetime.now(UTC).isoformat(),
        }
        async with AsyncSessionLocal() as db:
            await db.execute(_insert(ModelVersion).values(
                version_id=version_id,
                trained_at=datetime.now(UTC).replace(tzinfo=None),
                dataset_hash=None,
                holdout_metrics=holdout,
                status="candidate",
                promoted_at=None,
                rejected_reason=None,
            ))
            await db.execute(_insert(RetrainJob).values(
                job_id=job_id,
                triggered_at=datetime.now(UTC).replace(tzinfo=None),
                trigger_reason="retrain_trigger_event",
                dataset_size=int(dataset_size),
                status="completed",
                completed_at=datetime.now(UTC).replace(tzinfo=None),
                resulting_model_version_id=version_id,
            ))
            await db.commit()
        logger.info("Recorded candidate version=%s job=%s", version_id, job_id)
        return version_id, job_id

    async def _publish_candidate_ready(self, version_id: str, train_result: dict[str, Any]) -> None:
        """Publish MODEL_CANDIDATE_READY ke Redis agar experiment-orchestrator eval."""
        try:
            from shared.messaging import Channels
            payload = {
                "version_id": version_id,
                "holdout_metrics": train_result.get("train_metrics") or {},
                "candidate_path": train_result.get("candidate_path"),
                "n_samples": int(train_result.get("valid_samples", 0)),
                "trained_at": datetime.now(UTC).isoformat(),
            }
            # message_bus di-pass lewat constructor atau pakai global
            bus = getattr(self, "_message_bus", None)
            if bus is not None:
                await bus.publish(Channels.MODEL_CANDIDATE_READY, payload)
                logger.info("Published MODEL_CANDIDATE_READY: %s", version_id)
            else:
                logger.warning("No message_bus wired; skip publish CANDIDATE_READY")
        except Exception as exc:  # noqa: BLE001
            logger.warning("publish MODEL_CANDIDATE_READY failed: %s", exc)

    async def handle_retrain_trigger(self, msg: dict[str, Any]) -> None:
        """Handler untuk Channels.RETRAIN_TRIGGER dari loss-analyzer."""
        logger.info("RETRAIN_TRIGGER received: %s", msg)
        try:
            result = await self.run_full_retrain()
            logger.info("Event-triggered retrain complete: %s", result.get("ensemble", {}).get("status"))
        except Exception as exc:  # noqa: BLE001
            logger.error("Event-triggered retrain failed: %s", exc)

    @staticmethod
    def _backup_model(filename: str) -> None:
        """Backup model lama dengan timestamp sebelum diganti model baru."""
        src = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(src):
            return
        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(MODEL_DIR, "backups")
        os.makedirs(backup_dir, exist_ok=True)
        dst = os.path.join(backup_dir, f"{filename}.{ts}.bak")
        shutil.copy2(src, dst)
        logger.info("Model backed up: %s → %s", os.path.basename(src), os.path.basename(dst))

    def get_status(self) -> dict[str, Any]:
        """Status retrainer untuk health check dan /retrainer/status endpoint."""
        next_run: str | None = None
        if self._last_retrain:
            next_dt = self._last_retrain + timedelta(days=RETRAIN_INTERVAL_DAYS)
            next_run = next_dt.isoformat()

        return {
            "running": self._running,
            "is_retraining_now": self._is_running_now,
            "last_retrain": self._last_retrain.isoformat() if self._last_retrain else None,
            "next_scheduled_retrain": next_run,
            "retrain_interval_days": RETRAIN_INTERVAL_DAYS,
            "min_candles_gmm": MIN_CANDLES_FOR_GMM,
            "min_trades_mae_mfe": MIN_TRADES_FOR_MAE_MFE,
            "gmm_trained": self.regime_classifier.is_trained(),
            "mae_mfe_trained": self.mae_mfe_predictor.is_trained(),
            "last_result": self._last_result,
        }
