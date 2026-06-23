"""MLflow Service — thin wrapper around the MLflow tracking API.

Provides a clean interface for the rest of the application to:
  • Create experiments
  • Log parameters, metrics, and models
  • Retrieve best runs
"""

from __future__ import annotations

import logging
from typing import Any

import mlflow
from mlflow.entities import Run

from ml_experiment_orchestrator.config import settings

logger = logging.getLogger(__name__)


class MLflowService:
    """Wrapper around MLflow tracking operations."""

    def __init__(self) -> None:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        logger.info("MLflow tracking URI: %s", settings.mlflow_tracking_uri)

    def create_experiment(self, name: str) -> str:
        """Create or get an MLflow experiment by name.

        Returns:
            The experiment ID string.
        """
        experiment = mlflow.get_experiment_by_name(name)
        if experiment is not None:
            return experiment.experiment_id
        experiment_id = mlflow.create_experiment(name)
        logger.info("Created MLflow experiment '%s' (id=%s)", name, experiment_id)
        return experiment_id

    def start_run(
        self,
        experiment_id: str,
        run_name: str,
    ) -> Any:
        """Start a new MLflow run within the given experiment.

        Usage::

            svc = MLflowService()
            with svc.start_run(exp_id, "xgboost_run_1") as run:
                svc.log_params({"lr": 0.01})
                svc.log_metrics({"f1": 0.85})

        Returns:
            An ``mlflow.ActiveRun`` context manager.
        """
        return mlflow.start_run(
            experiment_id=experiment_id,
            run_name=run_name,
        )

    @staticmethod
    def log_params(params: dict[str, Any]) -> None:
        """Log parameters to the active MLflow run."""
        # MLflow requires string values for params
        sanitized = {k: str(v) for k, v in params.items()}
        mlflow.log_params(sanitized)

    @staticmethod
    def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics to the active MLflow run."""
        mlflow.log_metrics(metrics, step=step)

    @staticmethod
    def log_model(model: Any, artifact_path: str = "model") -> None:
        """Log a scikit-learn compatible model to the active run."""
        mlflow.sklearn.log_model(model, artifact_path=artifact_path)

    @staticmethod
    def get_active_run_id() -> str | None:
        """Return the current active run ID, or None."""
        run = mlflow.active_run()
        return run.info.run_id if run else None

    def get_best_run(
        self,
        experiment_id: str,
        metric: str,
        ascending: bool = False,
    ) -> Run | None:
        """Find the best run in an experiment by a given metric.

        Args:
            experiment_id: MLflow experiment ID.
            metric: Metric name to sort by.
            ascending: If True, lower is better (e.g., loss).

        Returns:
            The best MLflow Run, or None if no runs exist.
        """
        order = "ASC" if ascending else "DESC"
        runs = mlflow.search_runs(
            experiment_ids=[experiment_id],
            order_by=[f"metrics.{metric} {order}"],
            max_results=1,
            output_format="list",
        )
        if not runs:
            return None
        return runs[0]

    @staticmethod
    def end_run() -> None:
        """End the current active run if one exists."""
        mlflow.end_run()
