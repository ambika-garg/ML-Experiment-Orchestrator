from __future__ import annotations

import pytest

from ml_experiment_orchestrator.agents.planner import PlannerAgent
from ml_experiment_orchestrator.agents.report import ReportAgent
from ml_experiment_orchestrator.config import settings


def test_demo_mode_planner_does_not_require_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "demo_mode", True)
    monkeypatch.setattr(settings, "google_api_key", "")

    result = PlannerAgent().run(
        {"goal": "Predict whether a tumor is malignant from clinical features"}
    )

    plan = result["experiment_plan"]
    assert plan["problem_type"] == "classification"
    assert plan["target_column"] == "target"
    assert "random_forest" in plan["models"]


def test_live_mode_without_api_key_fails_with_demo_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "demo_mode", False)
    monkeypatch.setattr(settings, "google_api_key", "")

    with pytest.raises(RuntimeError, match="DEMO_MODE=true"):
        PlannerAgent().invoke_llm_json("User goal: predict cancer")


def test_demo_mode_report_is_markdown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "demo_mode", True)
    monkeypatch.setattr(settings, "google_api_key", "")

    result = ReportAgent().run(
        {
            "goal": "Predict diabetes progression",
            "experiment_plan": {"problem_type": "regression"},
            "dataset_summary": {"n_rows": 442, "n_cols": 11},
            "feature_plan": {},
            "experiment_results": [],
            "evaluation": {},
            "best_model": {"model_name": "random_forest", "metrics": {"rmse": 55.0}},
            "critic_feedback": {},
            "iteration": 0,
        }
    )

    report = result["final_report"]
    assert report.startswith("# ML Experiment Report")
    assert "demo mode" in report.lower()
