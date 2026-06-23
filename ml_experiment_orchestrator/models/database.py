"""SQLAlchemy ORM models for experiment persistence.

Two tables:
  • Experiment  — top-level record for each user goal
  • ExperimentRun — individual model training runs within an experiment
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    return str(uuid.uuid4())


class Experiment(Base):
    """Top-level experiment record corresponding to a user goal."""

    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_new_uuid
    )
    goal: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="pending"
    )  # pending | running | completed | failed

    # JSON blobs for experiment state
    experiment_plan: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    dataset_summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    feature_plan: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    best_model: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    final_report: Mapped[str | None] = mapped_column(Text, nullable=True)
    dataset_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    current_iteration: Mapped[int] = mapped_column(Integer, default=0)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    # Relationships
    runs: Mapped[list["ExperimentRun"]] = relationship(
        back_populates="experiment",
        cascade="all, delete-orphan",
        order_by="ExperimentRun.created_at",
    )

    def __repr__(self) -> str:
        return f"<Experiment {self.id[:8]}… status={self.status}>"


class ExperimentRun(Base):
    """Individual model training run within an experiment."""

    __tablename__ = "experiment_runs"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_new_uuid
    )
    experiment_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("experiments.id"), nullable=False
    )
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    parameters: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    mlflow_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    iteration: Mapped[int] = mapped_column(Integer, default=0)
    stage: Mapped[str] = mapped_column(
        String(20), default="initial"
    )  # initial | hpo | replan

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )

    # Relationships
    experiment: Mapped["Experiment"] = relationship(back_populates="runs")

    def __repr__(self) -> str:
        return f"<ExperimentRun {self.id[:8]}… model={self.model_name}>"
