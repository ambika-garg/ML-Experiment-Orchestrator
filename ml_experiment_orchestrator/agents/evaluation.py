"""Evaluation Agent — computes metrics, ranks models, and identifies the best performer.

Combines programmatic metric aggregation with LLM-driven interpretation
to produce a comprehensive evaluation of all experiment runs.
"""

from __future__ import annotations

import logging
from typing import Any

from ml_experiment_orchestrator.agents.base import BaseAgent

logger = logging.getLogger(__name__)


class EvaluationAgent(BaseAgent):
    """Evaluate and rank all experiment runs, identifying the best model."""

    @property
    def system_prompt(self) -> str:
        return """You are an expert ML model evaluator. Given a list of experiment results with metrics, you must:

1. Rank the models by performance
2. Identify the best model and explain why
3. Note any concerning patterns (overfitting signs, close competitors, etc.)
4. Provide a brief analysis of the results

Respond ONLY with a valid JSON object in this format:
{
  "rankings": [
    {"rank": 1, "model_name": "xgboost", "primary_metric_value": 0.92, "stage": "hpo"},
    {"rank": 2, "model_name": "random_forest", "primary_metric_value": 0.89, "stage": "initial"}
  ],
  "best_model": {
    "model_name": "xgboost",
    "metrics": {"f1": 0.90, "roc_auc": 0.92},
    "stage": "hpo",
    "mlflow_run_id": "abc123"
  },
  "analysis": "Brief analysis of results and patterns observed",
  "concerns": ["Any concerns about the results"]
}

Do NOT include any text outside the JSON object."""

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Evaluate all experiment results and identify the best model.

        Args:
            state: Must contain ``experiment_results`` and ``experiment_plan``.

        Returns:
            Partial state with ``evaluation`` and ``best_model``.
        """
        results = state.get("experiment_results", [])
        plan = state.get("experiment_plan", {})
        primary_metric = plan.get("primary_metric", "f1")

        logger.info(
            "[EvaluationAgent] Evaluating %d results on %s",
            len(results),
            primary_metric,
        )

        # ── Programmatic Ranking ──────────────────────────────────────────
        valid_results = [r for r in results if "error" not in r and r.get("metrics")]

        if not valid_results:
            logger.warning("[EvaluationAgent] No valid results to evaluate")
            return {
                "evaluation": {"rankings": [], "best_model": None, "analysis": "No valid results"},
                "best_model": {},
            }

        # Determine if higher or lower is better
        lower_is_better = primary_metric in ("mse", "rmse", "mae")

        # Sort results
        sorted_results = sorted(
            valid_results,
            key=lambda r: r.get("metrics", {}).get(primary_metric, 0),
            reverse=not lower_is_better,
        )

        # Build rankings table for LLM
        rankings_text = "Model Results (sorted by {}):\n".format(primary_metric)
        for i, r in enumerate(sorted_results, 1):
            rankings_text += (
                f"  {i}. {r['model_name']} ({r.get('stage', 'initial')}): "
                f"{r.get('metrics', {})}\n"
            )

        # ── LLM Analysis ─────────────────────────────────────────────────
        prompt = (
            f"Evaluate these ML experiment results.\n\n"
            f"Primary metric: {primary_metric} "
            f"({'lower is better' if lower_is_better else 'higher is better'})\n\n"
            f"{rankings_text}"
        )

        llm_eval = self.invoke_llm_json(prompt)

        # ── Build Evaluation ──────────────────────────────────────────────
        best = sorted_results[0]
        best_model = {
            "model_name": best["model_name"],
            "metrics": best.get("metrics", {}),
            "parameters": best.get("parameters", {}),
            "stage": best.get("stage", "initial"),
            "mlflow_run_id": best.get("mlflow_run_id"),
            "iteration": best.get("iteration", 0),
        }

        evaluation = {
            "rankings": [
                {
                    "rank": i + 1,
                    "model_name": r["model_name"],
                    "primary_metric_value": r.get("metrics", {}).get(primary_metric, 0),
                    "metrics": r.get("metrics", {}),
                    "stage": r.get("stage", "initial"),
                }
                for i, r in enumerate(sorted_results)
            ],
            "best_model": best_model,
            "analysis": llm_eval.get("analysis", ""),
            "concerns": llm_eval.get("concerns", []),
            "primary_metric": primary_metric,
            "primary_metric_value": best.get("metrics", {}).get(primary_metric, 0),
        }

        logger.info(
            "[EvaluationAgent] Best model: %s (%s=%.4f)",
            best_model["model_name"],
            primary_metric,
            best.get("metrics", {}).get(primary_metric, 0),
        )

        return {"evaluation": evaluation, "best_model": best_model}
