"""FastAPI application entry point.

Configures the app with:
  • Lifespan handler (DB init on startup)
  • CORS middleware
  • API router
  • Health check endpoint
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ml_experiment_orchestrator.api.routes import router
from ml_experiment_orchestrator.config import settings
from ml_experiment_orchestrator.db.session import init_db

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle handler."""
    logger.info("🚀 Starting ML Experiment Orchestrator")
    logger.info("   Database: %s", settings.database_url[:50] + "…")
    logger.info("   MLflow:   %s", settings.mlflow_tracking_uri)
    logger.info("   LLM:      %s", settings.llm_model)

    await init_db()
    logger.info("   Database tables initialised ✓")

    yield

    logger.info("👋 Shutting down ML Experiment Orchestrator")


# ── App Factory ───────────────────────────────────────────────────────────
def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="ML Experiment Orchestrator",
        description=(
            "Autonomous Multi-Agent ML Experiment Orchestrator — "
            "an AI Research Scientist that plans, executes, evaluates, "
            "and improves machine learning experiments."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS — allow all origins for development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routes
    app.include_router(router, prefix="/api/v1")

    # Health check
    @app.get("/health")
    async def health() -> dict:
        return {"status": "healthy", "version": "0.1.0"}

    return app


# ── Uvicorn entry point ──────────────────────────────────────────────────

app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "ml_experiment_orchestrator.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
