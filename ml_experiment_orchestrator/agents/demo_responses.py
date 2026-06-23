"""Deterministic agent responses for running the app without an LLM API key."""

from __future__ import annotations

import json
import re
from typing import Any


def demo_json_response(agent_name: str, prompt: str) -> dict[str, Any]:
    """Return a deterministic JSON response for an LLM-backed agent."""
    if agent_name == "PlannerAgent":
        return _planner_response(prompt)
    if agent_name == "DataAnalysisAgent":
        return {
            "quality_assessment": (
                "Demo analysis: the dataset is usable for an automated baseline "
                "experiment, with standard preprocessing recommended."
            ),
            "recommendations": [
                "Use numeric imputation before scaling.",
                "Compare simple linear models against tree-based ensembles.",
                "Track both the primary metric and secondary diagnostic metrics.",
            ],
            "potential_issues": [
                "Demo mode uses deterministic agent reasoning rather than live LLM analysis."
            ],
            "feature_importance_hints": [],
        }
    if agent_name == "FeatureEngineeringAgent":
        return {
            "preprocessing_steps": [
                {"step": "imputation", "method": "median", "columns": "numeric"},
                {"step": "imputation", "method": "most_frequent", "columns": "categorical"},
                {"step": "scaling", "method": "standard", "columns": "numeric"},
                {"step": "encoding", "method": "onehot", "columns": "categorical"},
            ],
            "feature_selection": {
                "apply": _shape_value(prompt, "Shape", default_cols=0) > 20,
                "method": "select_k_best",
                "n_features": 20,
            },
            "imbalance_handling": {"apply": False, "method": None},
            "reasoning": (
                "Demo mode applies a conservative preprocessing pipeline that works "
                "for the built-in datasets and most tabular CSVs."
            ),
        }
    if agent_name == "EvaluationAgent":
        return {
            "analysis": (
                "Demo evaluator: models were ranked by the configured primary metric. "
                "Use the MLflow run IDs to inspect individual training details."
            ),
            "concerns": [
                "This interpretation was generated in demo mode without an external LLM."
            ],
        }
    if agent_name == "CriticAgent":
        return {
            "should_continue": False,
            "improvements": [],
            "reasoning": (
                "Demo mode stops after the first evaluation to keep local runs fast "
                "and deterministic without requiring an API key."
            ),
            "confidence": 1.0,
        }
    if agent_name == "ReplannerAgent":
        return {
            "experiment_plan": _planner_response(prompt),
            "feature_plan": demo_json_response("FeatureEngineeringAgent", prompt),
        }
    return {}


def demo_text_response(agent_name: str, prompt: str) -> str:
    """Return deterministic plain-text output for an LLM-backed agent."""
    if agent_name == "ReportAgent":
        return _report_response(prompt)
    return json.dumps(demo_json_response(agent_name, prompt), indent=2)


def _planner_response(prompt: str) -> dict[str, Any]:
    text = prompt.lower()
    is_regression = any(
        token in text
        for token in ("regression", "diabetes", "price", "value", "score", "forecast")
    )

    if is_regression:
        return {
            "problem_type": "regression",
            "models": ["random_forest", "gradient_boosting", "xgboost"],
            "metrics": ["rmse", "mae", "r2"],
            "primary_metric": "rmse",
            "target_column": "target",
            "approach": (
                "Demo plan: compare tree-based regressors with standard numeric "
                "preprocessing and rank by RMSE."
            ),
        }

    return {
        "problem_type": "classification",
        "models": ["logistic_regression", "random_forest", "xgboost"],
        "metrics": ["f1", "roc_auc", "accuracy"],
        "primary_metric": "f1",
        "target_column": "target",
        "approach": (
            "Demo plan: compare a linear baseline with tree-based classifiers and "
            "rank by F1 score."
        ),
    }


def _report_response(prompt: str) -> str:
    goal = _extract_after(prompt, "User Goal:", default="Demo experiment")
    best_model = _extract_after(prompt, "Best Model:", default="N/A")
    best_metrics = _extract_after(prompt, "Best Metrics:", default="{}")

    return f"""# ML Experiment Report

## Executive Summary

This report was generated in demo mode, so agent reasoning is deterministic and no external LLM API key was used. The workflow still loaded data, trained models, evaluated metrics, and selected a best run.

## Objective

{goal}

## Methodology

The orchestrator created a tabular ML plan, profiled the dataset, applied a conservative preprocessing pipeline, trained candidate models, and ranked them with the configured primary metric.

## Best Model

**Model:** {best_model}

**Metrics:** `{best_metrics}`

## Notes

- Demo mode is intended for local walkthroughs, portfolio demos, and CI-friendly smoke tests.
- Set `DEMO_MODE=false` and provide `GOOGLE_API_KEY` to enable live Gemini-powered agent reasoning.
"""


def _extract_after(prompt: str, label: str, default: str) -> str:
    pattern = rf"{re.escape(label)}\s*(.+)"
    match = re.search(pattern, prompt)
    if not match:
        return default
    return match.group(1).strip()


def _shape_value(prompt: str, label: str, default_cols: int) -> int:
    pattern = rf"{re.escape(label)}:\s*\d+\s*[×x]\s*(\d+)"
    match = re.search(pattern, prompt)
    if match:
        return int(match.group(1))
    return default_cols
