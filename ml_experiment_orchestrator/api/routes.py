"""FastAPI route definitions for the experiment orchestrator API.

Endpoints:
  POST   /experiment           — Create and launch an experiment
  GET    /experiment/{id}      — Get experiment status and details
  GET    /experiment/{id}/report — Get the final Markdown report
  GET    /experiment/{id}/runs — List all training runs
  GET    /experiments          — List all experiments
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from ml_experiment_orchestrator.api.dependencies import get_experiment_service
from ml_experiment_orchestrator.models.schemas import (
    ExperimentCreate,
    ExperimentDetailResponse,
    ExperimentListResponse,
    ExperimentReportResponse,
    ExperimentResponse,
    ExperimentRunResponse,
)
from ml_experiment_orchestrator.services.experiment_service import ExperimentService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["experiments"])


# ── Background task helper ────────────────────────────────────────────────

async def _run_experiment_background(experiment_id: str) -> None:
    """Run an experiment in the background (called by BackgroundTasks)."""
    from ml_experiment_orchestrator.db.session import async_session_factory

    async with async_session_factory() as session:
        try:
            svc = ExperimentService(session)
            await svc.run_experiment(experiment_id)
            await session.commit()
        except Exception:
            await session.rollback()
            logger.exception("Background experiment %s failed", experiment_id)


# ── Routes ────────────────────────────────────────────────────────────────


@router.post("/experiment", response_model=ExperimentResponse, status_code=201)
async def create_experiment(
    payload: ExperimentCreate,
    background_tasks: BackgroundTasks,
    svc: ExperimentService = Depends(get_experiment_service),
) -> ExperimentResponse:
    """Create a new experiment and launch it as a background task.

    The experiment starts immediately in the background.  Poll
    ``GET /experiment/{id}`` to check progress.
    """
    experiment = await svc.create_experiment(
        goal=payload.goal,
        dataset=payload.dataset,
    )

    # Launch workflow in background
    background_tasks.add_task(_run_experiment_background, experiment.id)

    logger.info("Experiment %s created and queued", experiment.id)
    return ExperimentResponse.model_validate(experiment)


@router.get("/experiment/{experiment_id}", response_model=ExperimentDetailResponse)
async def get_experiment(
    experiment_id: str,
    svc: ExperimentService = Depends(get_experiment_service),
) -> ExperimentDetailResponse:
    """Get experiment details including all runs."""
    experiment = await svc.get_experiment(experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return ExperimentDetailResponse.model_validate(experiment)


@router.get(
    "/experiment/{experiment_id}/report", response_model=ExperimentReportResponse
)
async def get_experiment_report(
    experiment_id: str,
    svc: ExperimentService = Depends(get_experiment_service),
) -> ExperimentReportResponse:
    """Get the final Markdown report for a completed experiment."""
    experiment = await svc.get_experiment(experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    if experiment.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Experiment is {experiment.status}, not completed yet.",
        )
    return ExperimentReportResponse.model_validate(experiment)


@router.get(
    "/experiment/{experiment_id}/runs",
    response_model=list[ExperimentRunResponse],
)
async def get_experiment_runs(
    experiment_id: str,
    svc: ExperimentService = Depends(get_experiment_service),
) -> list[ExperimentRunResponse]:
    """List all training runs for an experiment."""
    runs = await svc.get_experiment_runs(experiment_id)
    return [ExperimentRunResponse.model_validate(r) for r in runs]


@router.get("/experiments", response_model=ExperimentListResponse)
async def list_experiments(
    limit: int = 50,
    offset: int = 0,
    svc: ExperimentService = Depends(get_experiment_service),
) -> ExperimentListResponse:
    """List all experiments with pagination."""
    experiments, total = await svc.list_experiments(limit=limit, offset=offset)
    return ExperimentListResponse(
        experiments=[ExperimentResponse.model_validate(e) for e in experiments],
        total=total,
    )
