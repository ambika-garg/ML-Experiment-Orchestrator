"""Experiment Service — high-level orchestration and database persistence.

Bridges the FastAPI layer and the LangGraph workflow, handling:
  • Experiment creation and persistence
  • Workflow invocation (synchronous and background)
  • State persistence to PostgreSQL/SQLite
  • Result retrieval
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ml_experiment_orchestrator.config import settings
from ml_experiment_orchestrator.graph.workflow import workflow
from ml_experiment_orchestrator.models.database import Experiment, ExperimentRun, LLMTrace
from ml_experiment_orchestrator.services.data_loader import load_dataset

logger = logging.getLogger(__name__)


class ExperimentService:
    """High-level service for creating, running, and querying experiments."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    # ── Create ────────────────────────────────────────────────────────────

    async def create_experiment(self, goal: str, dataset: str) -> Experiment:
        """Create a new experiment record in the database.

        Args:
            goal: Natural-language user goal.
            dataset: Dataset identifier or path.

        Returns:
            The newly created ``Experiment`` ORM instance.
        """
        experiment = Experiment(
            goal=goal,
            dataset_path=dataset,
            status="pending",
        )
        self.session.add(experiment)
        await self.session.commit()
        logger.info("Created experiment %s for goal: %s", experiment.id, goal)
        return experiment

    # ── Run ───────────────────────────────────────────────────────────────

    async def run_experiment(self, experiment_id: str) -> Experiment:
        """Execute the full LangGraph workflow for an experiment.

        This is a synchronous (blocking) execution suitable for background tasks.

        Args:
            experiment_id: The UUID of the experiment to run.

        Returns:
            The updated ``Experiment`` instance after workflow completion.
        """
        # Load experiment from DB
        experiment = await self._get_experiment(experiment_id)
        if experiment is None:
            raise ValueError(f"Experiment {experiment_id} not found")

        # Update status
        experiment.status = "running"
        await self.session.flush()

        try:
            # Load dataset
            df = load_dataset(experiment.dataset_path or "breast_cancer")

            # Build initial state
            initial_state = {
                "goal": experiment.goal,
                "dataset_path": experiment.dataset_path or "breast_cancer",
                "dataset": df,
                "experiment_id": experiment.id,
                "iteration": 0,
                "max_iterations": settings.max_iterations,
                "experiment_results": [],
            }

            # Run the LangGraph workflow in a thread pool so it doesn't block the async event loop
            import asyncio
            logger.info("Starting workflow for experiment %s", experiment_id)
            final_state = await asyncio.to_thread(workflow.invoke, initial_state)

            # Persist results to database
            await self._persist_results(experiment, final_state)

            experiment.status = "completed"
            logger.info("Experiment %s completed successfully", experiment_id)

        except Exception as exc:
            logger.exception("Experiment %s failed: %s", experiment_id, exc)
            experiment.status = "failed"
            experiment.final_report = f"Experiment failed: {exc}"

        await self.session.flush()
        return experiment

    # ── Query ─────────────────────────────────────────────────────────────

    async def get_experiment(self, experiment_id: str) -> Experiment | None:
        """Get an experiment by ID with its runs and LLM traces eagerly loaded."""
        stmt = (
            select(Experiment)
            .where(Experiment.id == experiment_id)
            .options(
                selectinload(Experiment.runs),
                selectinload(Experiment.llm_traces),
            )
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_experiments(
        self, limit: int = 50, offset: int = 0
    ) -> tuple[list[Experiment], int]:
        """List all experiments with pagination.

        Returns:
            Tuple of (experiments list, total count).
        """
        # Count
        from sqlalchemy import func

        count_stmt = select(func.count()).select_from(Experiment)
        count_result = await self.session.execute(count_stmt)
        total = count_result.scalar_one()

        # Fetch
        stmt = (
            select(Experiment)
            .order_by(Experiment.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(stmt)
        experiments = list(result.scalars().all())
        return experiments, total

    async def get_experiment_runs(
        self, experiment_id: str
    ) -> list[ExperimentRun]:
        """Get all runs for an experiment."""
        stmt = (
            select(ExperimentRun)
            .where(ExperimentRun.experiment_id == experiment_id)
            .order_by(ExperimentRun.created_at)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    # ── Private ───────────────────────────────────────────────────────────

    async def _get_experiment(self, experiment_id: str) -> Experiment | None:
        """Get experiment without eager loading."""
        stmt = select(Experiment).where(Experiment.id == experiment_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def _persist_results(
        self, experiment: Experiment, state: dict[str, Any]
    ) -> None:
        """Persist the final workflow state to the database."""
        # Update experiment fields
        experiment.experiment_plan = state.get("experiment_plan")
        experiment.dataset_summary = self._make_serializable(
            state.get("dataset_summary", {})
        )
        experiment.feature_plan = state.get("feature_plan")
        experiment.best_model = state.get("best_model")
        experiment.final_report = state.get("final_report")
        experiment.current_iteration = state.get("iteration", 0)

        # Create ExperimentRun records
        for result in state.get("experiment_results", []):
            run = ExperimentRun(
                experiment_id=experiment.id,
                model_name=result.get("model_name", "unknown"),
                parameters=result.get("parameters"),
                metrics=result.get("metrics"),
                mlflow_run_id=result.get("mlflow_run_id"),
                iteration=result.get("iteration", 0),
                stage=result.get("stage", "initial"),
            )
            self.session.add(run)

        # Create LLMTrace records
        for trace in state.get("llm_traces", []):
            db_trace = LLMTrace(
                experiment_id=experiment.id,
                agent_name=trace.get("agent_name", "unknown"),
                prompt_tokens=trace.get("prompt_tokens", 0),
                completion_tokens=trace.get("completion_tokens", 0),
                total_tokens=trace.get("total_tokens", 0),
                cost=trace.get("cost", 0.0),
                latency=trace.get("latency", 0.0),
            )
            self.session.add(db_trace)

        await self.session.flush()
        logger.info(
            "Persisted %d runs and %d LLM traces for experiment %s",
            len(state.get("experiment_results", [])),
            len(state.get("llm_traces", [])),
            experiment.id,
        )

    @staticmethod
    def _make_serializable(data: Any) -> Any:
        """Ensure data is JSON-serializable by converting numpy types."""
        import numpy as np

        if isinstance(data, dict):
            return {
                k: ExperimentService._make_serializable(v)
                for k, v in data.items()
            }
        if isinstance(data, list):
            return [ExperimentService._make_serializable(v) for v in data]
        if isinstance(data, (np.integer,)):
            return int(data)
        if isinstance(data, (np.floating,)):
            return float(data)
        if isinstance(data, np.ndarray):
            return data.tolist()
        return data
