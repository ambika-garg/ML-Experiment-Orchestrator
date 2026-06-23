"""Pydantic v2 schemas for API request / response serialization."""

from datetime import datetime

from pydantic import BaseModel, Field


# ── Request Schemas ───────────────────────────────────────────────────────


class ExperimentCreate(BaseModel):
    """Payload for creating a new experiment."""

    goal: str = Field(
        ...,
        min_length=5,
        examples=["Predict heart disease from patient clinical data"],
    )
    dataset: str = Field(
        default="heart_disease",
        description=(
            "Dataset identifier: 'heart_disease', 'diabetes', 'breast_cancer' "
            "(built-in), or an absolute path to a CSV file."
        ),
    )


# ── Response Schemas ──────────────────────────────────────────────────────


class ExperimentRunResponse(BaseModel):
    """Single training run within an experiment."""

    id: str
    model_name: str
    parameters: dict | None = None
    metrics: dict | None = None
    mlflow_run_id: str | None = None
    iteration: int
    stage: str
    created_at: datetime

    model_config = {"from_attributes": True}


class ExperimentResponse(BaseModel):
    """Summary view of an experiment."""

    id: str
    goal: str
    status: str
    experiment_plan: dict | None = None
    dataset_summary: dict | None = None
    feature_plan: dict | None = None
    best_model: dict | None = None
    dataset_path: str | None = None
    current_iteration: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class LLMTraceResponse(BaseModel):
    """Single LLM API call trace information."""

    id: str
    agent_name: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float
    latency: float
    created_at: datetime

    model_config = {"from_attributes": True}


class ExperimentDetailResponse(ExperimentResponse):
    """Experiment with its runs and LLM traces attached."""

    runs: list[ExperimentRunResponse] = []
    llm_traces: list[LLMTraceResponse] = []


class ExperimentReportResponse(BaseModel):
    """Just the final Markdown report."""

    id: str
    goal: str
    status: str
    final_report: str | None = None

    model_config = {"from_attributes": True}


class ExperimentListResponse(BaseModel):
    """Paginated list of experiments."""

    experiments: list[ExperimentResponse]
    total: int
