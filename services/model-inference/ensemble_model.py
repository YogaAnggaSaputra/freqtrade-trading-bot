"""
ensemble_model.py
==================
Multi-Model Stacking Ensemble — Institutional-Grade ML Inference

Menggabungkan tiga model berbeda menggunakan Stacking Ensemble untuk
menghasilkan prediksi yang lebih robust dan tahan overfitting:

Base Models (Level 0):
  - XGBoost   : Gradient boosting, kuat untuk data tabular non-linear
  - LightGBM  : Lebih cepat dari XGBoost, cocok untuk high-cardinality features
  - CatBoost  : Robust terhadap categorical features & outliers

Meta-Learner (Level 1):
  - LogisticRegression : Kombinasi linearly dari prediksi base models

SHAP Integration:
  - Menjelaskan kontribusi setiap feature pada setiap prediksi
  - Membantu loss analyzer & Hermes agent memahami kenapa bot melakukan trade

Uncertainty Estimation:
  - std_dev dari base model probabilities → model uncertainty
  - Entropy Shannon dari prediksi → distributional uncertainty
  - Kedua metric digunakan oleh UncertaintyFilter untuk blokir low-confidence trades

Fallback cascade:
  XGBoost + LightGBM + CatBoost → XGBoost only → rule-based
"""
from __future__ import annotations

import contextlib
import logging
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("model_inference.ensemble_model")

MODEL_DIR = os.getenv("MODEL_DIR", "/models")
ENSEMBLE_MODEL_PATH = os.path.join(MODEL_DIR, "ensemble_stacking.pkl")


def _try_import_lgbm():
    try:
        import lightgbm as lgb
        return lgb
    except ImportError:
        return None


def _try_import_catboost():
    try:
        from catboost import CatBoostClassifier
        return CatBoostClassifier
    except ImportError:
        return None


def _try_import_xgb():
    try:
        import xgboost as xgb
        return xgb
    except ImportError:
        return None


def _try_import_shap():
    try:
        import shap
        return shap
    except ImportError:
        return None


class StackingEnsemble:
    """
    Multi-model stacking classifier dengan SHAP interpretability.

    Training:
        ensemble = StackingEnsemble()
        ensemble.fit(X_train, y_train)
        ensemble.save()

    Inference:
        result = ensemble.predict_proba_with_uncertainty(X)
        # → {'probability': 0.73, 'std_dev': 0.08, 'entropy': 0.52, 'base_probs': {...}}
    """

    def __init__(self):
        self._base_models: dict[str, Any] = {}
        self._meta_learner: Any | None = None
        self._shap_explainer: Any | None = None
        self._feature_names: list[str] = []
        self._is_trained: bool = False
        self._available_models: list[str] = []

    def is_trained(self) -> bool:
        return self._is_trained and self._meta_learner is not None

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str] | None = None) -> dict[str, Any]:
        """
        Latih stacking ensemble dari training data.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Binary labels (0 = no signal, 1 = signal)
            feature_names: Optional list of feature names untuk SHAP

        Returns:
            Training summary dict
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_predict

        if feature_names:
            self._feature_names = feature_names

        n_samples = len(X)
        logger.info("Training StackingEnsemble with %d samples, %d features", n_samples, X.shape[1])

        # ── Build available base models ────────────────────────────────────────
        base_preds = {}  # model_name → oof_predictions (out-of-fold)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # XGBoost
        xgb_mod = _try_import_xgb()
        if xgb_mod:
            try:
                model = xgb_mod.XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    use_label_encoder=False,
                    eval_metric="logloss",
                    random_state=42,
                    verbosity=0,
                )
                oof = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
                model.fit(X, y)
                self._base_models["xgboost"] = model
                base_preds["xgboost"] = oof
                self._available_models.append("xgboost")
                logger.info("XGBoost trained successfully")
            except Exception as e:
                logger.warning("XGBoost training failed: %s", e)

        # LightGBM
        lgb_mod = _try_import_lgbm()
        if lgb_mod:
            try:
                model = lgb_mod.LGBMClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbose=-1,
                )
                oof = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
                model.fit(X, y)
                self._base_models["lightgbm"] = model
                base_preds["lightgbm"] = oof
                self._available_models.append("lightgbm")
                logger.info("LightGBM trained successfully")
            except Exception as e:
                logger.warning("LightGBM training failed: %s", e)

        # CatBoost
        CatBoost = _try_import_catboost()
        if CatBoost:
            try:
                model = CatBoost(
                    iterations=300,
                    learning_rate=0.05,
                    depth=6,
                    random_seed=42,
                    verbose=False,
                )
                oof = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
                model.fit(X, y)
                self._base_models["catboost"] = model
                base_preds["catboost"] = oof
                self._available_models.append("catboost")
                logger.info("CatBoost trained successfully")
            except Exception as e:
                logger.warning("CatBoost training failed: %s", e)

        if not self._base_models:
            raise RuntimeError("No base models could be trained. Install xgboost, lightgbm, or catboost.")

        # ── Train Meta-Learner ─────────────────────────────────────────────────
        meta_X = np.column_stack(list(base_preds.values()))
        self._meta_learner = LogisticRegression(C=1.0, random_state=42)
        self._meta_learner.fit(meta_X, y)
        logger.info("Meta-learner (LogisticRegression) trained on %d base model OOF predictions", len(base_preds))

        # ── SHAP Explainer (using XGBoost if available) ─────────────────────
        shap_mod = _try_import_shap()
        if shap_mod and "xgboost" in self._base_models:
            try:
                self._shap_explainer = shap_mod.TreeExplainer(self._base_models["xgboost"])
                logger.info("SHAP TreeExplainer initialized")
            except Exception as e:
                logger.warning("SHAP explainer init failed: %s", e)

        self._is_trained = True

        return {
            "status": "success",
            "available_models": self._available_models,
            "n_samples": n_samples,
            "meta_learner": "LogisticRegression",
            "shap_available": self._shap_explainer is not None,
        }

    def predict_proba_with_uncertainty(self, X: np.ndarray) -> dict[str, Any]:
        """
        Prediksi probability dengan uncertainty estimation.

        Returns:
            {
              'probability': float,        # Meta-learner final probability
              'std_dev': float,            # Std deviation of base model probabilities
              'entropy': float,            # Shannon entropy (uncertainty measure)
              'base_probs': dict,          # Per-model probabilities
              'ensemble_available': bool
            }
        """
        if not self.is_trained():
            return {
                "probability": 0.5,
                "std_dev": 0.5,
                "entropy": 1.0,
                "base_probs": {},
                "ensemble_available": False,
                "reason": "Ensemble not trained",
            }

        base_probs = {}
        for name, model in self._base_models.items():
            try:
                prob = float(model.predict_proba(X.reshape(1, -1))[0, 1])
                base_probs[name] = prob
            except Exception as e:
                logger.warning("Base model %s prediction failed: %s", name, e)

        if not base_probs:
            return {
                "probability": 0.5,
                "std_dev": 0.5,
                "entropy": 1.0,
                "base_probs": {},
                "ensemble_available": False,
                "reason": "All base models failed",
            }

        # Meta-learner prediction
        meta_X = np.array([[base_probs.get(name, 0.5) for name in self._available_models]])
        try:
            final_prob = float(self._meta_learner.predict_proba(meta_X)[0, 1])
        except Exception:
            final_prob = float(np.mean(list(base_probs.values())))

        probs_array = np.array(list(base_probs.values()))
        std_dev = float(np.std(probs_array))

        # Shannon entropy: H = -Σ p*log(p) — normalized [0, 1]
        p = np.clip([final_prob, 1 - final_prob], 1e-9, 1 - 1e-9)
        entropy = float(-np.sum(p * np.log2(p)) / np.log2(len(p)))

        return {
            "probability": final_prob,
            "std_dev": std_dev,
            "entropy": entropy,
            "base_probs": base_probs,
            "ensemble_available": True,
        }

    def get_shap_explanation(self, X: np.ndarray, top_k: int = 10) -> dict[str, float]:
        """
        Dapatkan SHAP feature importance untuk satu sample.
        Returns top_k features dengan nilai SHAP tertinggi.
        """
        if self._shap_explainer is None or not self._feature_names:
            return {}

        try:
            shap_values = self._shap_explainer.shap_values(X.reshape(1, -1))
            if isinstance(shap_values, list):
                sv = shap_values[1][0]  # class 1
            else:
                sv = shap_values[0]

            importance = {
                self._feature_names[i]: float(sv[i])
                for i in range(min(len(self._feature_names), len(sv)))
            }
            # Sort by absolute value
            sorted_imp = dict(sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k])
            return sorted_imp
        except Exception as e:
            logger.warning("SHAP explanation failed: %s", e)
            return {}

    def save(self, path: str | None = None) -> str:
        """Simpan model ke disk. Jika path diberikan, simpan ke path tersebut
        (untuk candidate model); default ke ENSEMBLE_MODEL_PATH (production)."""
        target = path or ENSEMBLE_MODEL_PATH
        Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as f:
            pickle.dump({
                "base_models": self._base_models,
                "meta_learner": self._meta_learner,
                "feature_names": self._feature_names,
                "available_models": self._available_models,
            }, f)
        logger.info("Ensemble model saved to %s", target)
        return target

    def load(self) -> bool:
        """Load model dari disk. Returns False jika tidak ada."""
        if not os.path.exists(ENSEMBLE_MODEL_PATH):
            logger.info("No ensemble model found at %s", ENSEMBLE_MODEL_PATH)
            return False
        try:
            with open(ENSEMBLE_MODEL_PATH, "rb") as f:
                data = pickle.load(f)
            self._base_models = data["base_models"]
            self._meta_learner = data["meta_learner"]
            self._feature_names = data.get("feature_names", [])
            self._available_models = data.get("available_models", list(self._base_models.keys()))
            self._is_trained = True

            # Re-init SHAP explainer
            shap_mod = _try_import_shap()
            if shap_mod and "xgboost" in self._base_models:
                with contextlib.suppress(Exception):
                    self._shap_explainer = shap_mod.TreeExplainer(self._base_models["xgboost"])

            logger.info("Ensemble model loaded. Base models: %s", self._available_models)
            return True
        except Exception as e:
            logger.error("Failed to load ensemble model: %s", e)
            return False
