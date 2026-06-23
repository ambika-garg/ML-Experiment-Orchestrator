"""FastAPI dependencies — database sessions, services."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from ml_experiment_orchestrator.services.experiment_service import ExperimentService


async def get_experiment_service() -> AsyncGenerator[ExperimentService, None]:
    """Provide an ExperimentService wired with a DB session.

    This is a FastAPI dependency — use with ``Depends(get_experiment_service)``.
    """
    from ml_experiment_orchestrator.db.session import async_session_factory

    async with async_session_factory() as session:
        try:
            svc = ExperimentService(session)
            yield svc
            await session.commit()
        except Exception:
            await session.rollback()
            raise
