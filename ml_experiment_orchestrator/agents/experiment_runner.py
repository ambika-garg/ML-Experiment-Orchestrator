"""Experiment Runner Agent — trains candidate models and logs results to MLflow.

This is a *procedural* agent (no LLM calls).  It:
  1. Applies the feature engineering pipeline to the dataset
  2. Trains each candidate model (with optional parallelism)
  3. Evaluates on a held-out test set
  4. Logs everything to MLflow
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

from ml_experiment_orchestrator.config import settings
from ml_experiment_orchestrator.services.mlflow_service import MLflowService
from ml_experiment_orchestrator.services.model_registry import get_model

logger = logging.getLogger(__name__)


class ExperimentRunnerAgent:
    """Execute model training runs with preprocessing and MLflow tracking."""

    def __init__(self) -> None:
        self.mlflow = MLflowService()

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Train all candidate models and return results.

        Args:
            state: Must contain ``dataset``, ``experiment_plan``, ``feature_plan``.
                   Optionally contains ``experiment_id``.

        Returns:
            Partial state with ``experiment_results`` (list of run dicts).
        """
        df: pd.DataFrame = state["dataset"]
        plan = state["experiment_plan"]
        feature_plan = state["feature_plan"]
        experiment_id = state.get("experiment_id", "default")
        iteration = state.get("iteration", 0)

        problem_type = plan.get("problem_type", "classification")
        target_col = plan.get("target_column", "target")
        metrics_list = plan.get("metrics", ["f1", "roc_auc"])
        models_to_train = plan.get("models", ["random_forest"])

        logger.info(
            "[ExperimentRunner] Iteration %d — training %d models: %s",
            iteration,
            len(models_to_train),
            models_to_train,
        )

        # ── Prepare Data ──────────────────────────────────────────────────
        X, y = self._prepare_data(df, target_col)
        X_processed, y_processed = self._apply_preprocessing(
            X, y, feature_plan, problem_type
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X_processed,
            y_processed,
            test_size=settings.test_size,
            random_state=settings.random_state,
            stratify=y_processed if problem_type == "classification" else None,
        )

        # ── Create MLflow Experiment ──────────────────────────────────────
        mlflow_experiment_id = self.mlflow.create_experiment(
            f"orchestrator_{experiment_id}"
        )

        # ── Train Models ──────────────────────────────────────────────────
        results = []
        for model_name in models_to_train:
            try:
                result = self._train_single_model(
                    model_name=model_name,
                    problem_type=problem_type,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    metrics_list=metrics_list,
                    mlflow_experiment_id=mlflow_experiment_id,
                    iteration=iteration,
                )
                results.append(result)
            except Exception as exc:
                logger.error(
                    "[ExperimentRunner] Failed to train %s: %s", model_name, exc
                )
                results.append(
                    {
                        "model_name": model_name,
                        "parameters": {},
                        "metrics": {},
                        "mlflow_run_id": None,
                        "iteration": iteration,
                        "stage": "initial",
                        "error": str(exc),
                    }
                )

        logger.info(
            "[ExperimentRunner] Completed %d/%d models",
            sum(1 for r in results if "error" not in r),
            len(models_to_train),
        )
        return {"experiment_results": results}

    # ── Private Methods ───────────────────────────────────────────────────

    def _prepare_data(
        self, df: pd.DataFrame, target_col: str
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Split DataFrame into features and target."""
        if target_col not in df.columns:
            # Try common names
            for candidate in ["target", "Target", "label", "Label", "class", "Class"]:
                if candidate in df.columns:
                    target_col = candidate
                    break
            else:
                # Use the last column as target
                target_col = df.columns[-1]
                logger.warning("Target column not found, using last column: %s", target_col)

        y = df[target_col].copy()
        X = df.drop(columns=[target_col]).copy()

        # Encode target if it's string-typed
        if y.dtype == object:
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y), name=target_col)

        return X, y

    def _apply_preprocessing(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_plan: dict,
        problem_type: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply the preprocessing pipeline from the feature plan."""
        X = X.copy()

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        # ── Preprocessing Steps ───────────────────────────────────────────
        for step in feature_plan.get("preprocessing_steps", []):
            step_type = step.get("step", "")
            method = step.get("method", "")

            if step_type == "imputation":
                if step.get("columns") == "numeric" or method in ("mean", "median"):
                    if numeric_cols:
                        if method == "mean":
                            X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
                        else:
                            X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
                elif step.get("columns") == "categorical" or method == "most_frequent":
                    if categorical_cols:
                        for col in categorical_cols:
                            X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else "unknown")

            elif step_type == "scaling":
                if numeric_cols:
                    scaler_map = {
                        "standard": StandardScaler,
                        "minmax": MinMaxScaler,
                        "robust": RobustScaler,
                    }
                    scaler_cls = scaler_map.get(method, StandardScaler)
                    scaler = scaler_cls()
                    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

            elif step_type == "encoding":
                if categorical_cols:
                    if method == "onehot":
                        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
                    else:
                        for col in categorical_cols:
                            le = LabelEncoder()
                            X[col] = le.fit_transform(X[col].astype(str))

        # Fill any remaining NaNs
        X = X.fillna(0)

        # ── Feature Selection ─────────────────────────────────────────────
        fs = feature_plan.get("feature_selection", {})
        if fs.get("apply"):
            method = fs.get("method", "select_k_best")
            n_features = fs.get("n_features") or min(20, X.shape[1])

            if method in ("select_k_best", "SelectKBest"):
                score_func = f_classif if problem_type == "classification" else f_regression
                selector = SelectKBest(score_func=score_func, k=min(n_features, X.shape[1]))
                X_arr = selector.fit_transform(X.values, y.values)
                X = pd.DataFrame(X_arr)
            elif method in ("pca", "PCA"):
                pca = PCA(n_components=min(n_features, X.shape[1]))
                X_arr = pca.fit_transform(X.values)
                X = pd.DataFrame(X_arr)

        X_out = X.values.astype(np.float64)
        y_out = y.values.astype(np.float64)

        # ── Imbalance Handling ────────────────────────────────────────────
        ih = feature_plan.get("imbalance_handling", {})
        if ih.get("apply") and problem_type == "classification":
            method = ih.get("method", "smote")
            try:
                if method == "smote":
                    sampler = SMOTE(random_state=settings.random_state)
                else:
                    sampler = RandomOverSampler(random_state=settings.random_state)
                X_out, y_out = sampler.fit_resample(X_out, y_out)
                logger.info("Applied %s: %d → %d samples", method, len(y.values), len(y_out))
            except Exception as exc:
                logger.warning("Imbalance handling failed (%s): %s", method, exc)

        return X_out, y_out

    def _train_single_model(
        self,
        model_name: str,
        problem_type: str,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        metrics_list: list[str],
        mlflow_experiment_id: str,
        iteration: int,
    ) -> dict[str, Any]:
        """Train a single model, evaluate, and log to MLflow."""
        model = get_model(model_name, problem_type)
        run_name = f"{model_name}_iter{iteration}"

        with self.mlflow.start_run(mlflow_experiment_id, run_name):
            # Train
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Compute metrics
            metrics = self._compute_metrics(
                y_test, y_pred, model, X_test, problem_type, metrics_list
            )

            # Log to MLflow
            params = model.get_params()
            # Filter out non-serializable params
            safe_params = {k: v for k, v in params.items() if isinstance(v, (str, int, float, bool, type(None)))}
            self.mlflow.log_params(safe_params)
            self.mlflow.log_metrics(metrics)
            self.mlflow.log_params({
                "model_name": model_name,
                "iteration": str(iteration),
                "problem_type": problem_type,
            })

            run_id = self.mlflow.get_active_run_id()
            logger.info(
                "[ExperimentRunner] %s → %s (run_id=%s)",
                model_name,
                {k: round(v, 4) for k, v in metrics.items()},
                run_id,
            )

        return {
            "model_name": model_name,
            "parameters": safe_params,
            "metrics": metrics,
            "mlflow_run_id": run_id,
            "iteration": iteration,
            "stage": "initial" if iteration == 0 else "replan",
        }

    @staticmethod
    def _compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model: Any,
        X_test: np.ndarray,
        problem_type: str,
        metrics_list: list[str],
    ) -> dict[str, float]:
        """Compute requested evaluation metrics."""
        result: dict[str, float] = {}

        if problem_type == "classification":
            # Determine averaging
            n_classes = len(set(y_true))
            avg = "binary" if n_classes == 2 else "weighted"

            for m in metrics_list:
                try:
                    if m == "accuracy":
                        result[m] = float(accuracy_score(y_true, y_pred))
                    elif m == "f1":
                        result[m] = float(f1_score(y_true, y_pred, average=avg, zero_division=0))
                    elif m == "precision":
                        result[m] = float(precision_score(y_true, y_pred, average=avg, zero_division=0))
                    elif m == "recall":
                        result[m] = float(recall_score(y_true, y_pred, average=avg, zero_division=0))
                    elif m == "roc_auc":
                        if hasattr(model, "predict_proba"):
                            y_proba = model.predict_proba(X_test)
                            if n_classes == 2:
                                result[m] = float(roc_auc_score(y_true, y_proba[:, 1]))
                            else:
                                result[m] = float(
                                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
                                )
                        else:
                            result[m] = 0.0
                except Exception as exc:
                    logger.warning("Metric %s failed: %s", m, exc)
                    result[m] = 0.0
        else:
            for m in metrics_list:
                try:
                    if m in ("mse", "mean_squared_error"):
                        result["mse"] = float(mean_squared_error(y_true, y_pred))
                    elif m in ("rmse", "root_mean_squared_error"):
                        result["rmse"] = float(mean_squared_error(y_true, y_pred, squared=False))
                    elif m == "r2":
                        result[m] = float(r2_score(y_true, y_pred))
                    elif m in ("mae", "mean_absolute_error"):
                        result["mae"] = float(mean_absolute_error(y_true, y_pred))
                except Exception as exc:
                    logger.warning("Metric %s failed: %s", m, exc)
                    result[m] = 0.0

        return result
