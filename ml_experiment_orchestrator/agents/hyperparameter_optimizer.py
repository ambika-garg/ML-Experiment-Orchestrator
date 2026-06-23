"""Hyperparameter Optimization Agent — uses Optuna for automated tuning.

A procedural agent (no LLM) that runs Bayesian hyperparameter search using
Optuna's TPE sampler with cross-validation scoring and MLflow logging.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import optuna
from sklearn.model_selection import cross_val_score

from ml_experiment_orchestrator.config import settings
from ml_experiment_orchestrator.services.mlflow_service import MLflowService
from ml_experiment_orchestrator.services.model_registry import (
    get_model,
    get_search_space,
)

logger = logging.getLogger(__name__)

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


class HyperparameterOptimizationAgent:
    """Run Optuna hyperparameter optimization for the best-performing models."""

    def __init__(self) -> None:
        self.mlflow = MLflowService()

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Optimize hyperparameters for top models from initial training.

        Args:
            state: Must contain ``experiment_results``, ``experiment_plan``,
                   and preprocessed training data context.

        Returns:
            Partial state appending HPO results to ``experiment_results``.
        """
        plan = state["experiment_plan"]
        results = state.get("experiment_results", [])
        experiment_id = state.get("experiment_id", "default")
        iteration = state.get("iteration", 0)

        problem_type = plan.get("problem_type", "classification")
        primary_metric = plan.get("primary_metric", "f1")
        n_trials = settings.hpo_n_trials

        # Determine scoring direction
        maximize_metrics = {"f1", "roc_auc", "accuracy", "precision", "recall", "r2"}
        direction = "maximize" if primary_metric in maximize_metrics else "minimize"
        scoring = self._metric_to_sklearn_scoring(primary_metric, problem_type)

        # Select top model from current iteration results
        current_results = [r for r in results if r.get("iteration") == iteration and "error" not in r]
        if not current_results:
            logger.warning("[HPO] No valid results to optimize. Skipping.")
            return {"experiment_results": []}

        # Sort by primary metric
        best_result = max(
            current_results,
            key=lambda r: r.get("metrics", {}).get(primary_metric, 0),
        )
        model_name = best_result["model_name"]

        logger.info(
            "[HPO] Optimizing %s for %s (%d trials)",
            model_name,
            primary_metric,
            n_trials,
        )

        # We need the preprocessed data from the runner
        # Access it from the state (set by the runner)
        X_train = state.get("_X_train")
        y_train = state.get("_y_train")
        X_test = state.get("_X_test")
        y_test = state.get("_y_test")

        if X_train is None:
            logger.warning("[HPO] No training data available. Skipping optimization.")
            return {"experiment_results": []}

        # ── Optuna Study ──────────────────────────────────────────────────
        def objective(trial: optuna.Trial) -> float:
            params = get_search_space(model_name, trial)
            model = get_model(model_name, problem_type, params)
            try:
                scores = cross_val_score(
                    model,
                    X_train,
                    y_train,
                    cv=min(settings.cv_folds, 5),
                    scoring=scoring,
                    n_jobs=-1,
                )
                return float(np.mean(scores))
            except Exception as exc:
                logger.debug("[HPO] Trial failed: %s", exc)
                return 0.0 if direction == "maximize" else float("inf")

        study = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=settings.random_state),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        # ── Train Final Model with Best Params ────────────────────────────
        best_params = study.best_params
        best_model = get_model(model_name, problem_type, best_params)
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)

        # Compute metrics using the runner's metric computation
        from ml_experiment_orchestrator.agents.experiment_runner import ExperimentRunnerAgent

        metrics = ExperimentRunnerAgent._compute_metrics(
            y_test, y_pred, best_model, X_test, problem_type, plan.get("metrics", [])
        )

        # Log to MLflow
        mlflow_experiment_id = self.mlflow.create_experiment(
            f"orchestrator_{experiment_id}"
        )
        run_name = f"{model_name}_hpo_iter{iteration}"
        with self.mlflow.start_run(mlflow_experiment_id, run_name):
            safe_params = {k: v for k, v in best_params.items() if isinstance(v, (str, int, float, bool, type(None)))}
            self.mlflow.log_params(safe_params)
            self.mlflow.log_params({
                "model_name": model_name,
                "iteration": str(iteration),
                "stage": "hpo",
                "n_trials": str(n_trials),
            })
            self.mlflow.log_metrics(metrics)
            run_id = self.mlflow.get_active_run_id()

        hpo_result = {
            "model_name": model_name,
            "parameters": best_params,
            "metrics": metrics,
            "mlflow_run_id": run_id,
            "iteration": iteration,
            "stage": "hpo",
            "n_trials": n_trials,
            "best_trial_value": study.best_value,
        }

        logger.info(
            "[HPO] Best %s=%s with params %s",
            primary_metric,
            metrics.get(primary_metric, "N/A"),
            best_params,
        )

        return {"experiment_results": [hpo_result]}

    @staticmethod
    def _metric_to_sklearn_scoring(metric: str, problem_type: str) -> str:
        """Map our metric names to sklearn scoring strings."""
        mapping = {
            "f1": "f1_weighted",
            "roc_auc": "roc_auc_ovr_weighted" if problem_type == "classification" else "r2",
            "accuracy": "accuracy",
            "precision": "precision_weighted",
            "recall": "recall_weighted",
            "r2": "r2",
            "mse": "neg_mean_squared_error",
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
        }
        return mapping.get(metric, "accuracy")
