"""Planner Agent — understands user goals and creates experiment plans.

Given a natural-language goal the Planner determines:
  • problem type (classification / regression)
  • candidate algorithms
  • evaluation metrics
  • target column hints
"""

from __future__ import annotations

import logging
from typing import Any

from ml_experiment_orchestrator.agents.base import BaseAgent

logger = logging.getLogger(__name__)


class PlannerAgent(BaseAgent):
    """Analyse a user goal and produce a structured experiment plan."""

    @property
    def system_prompt(self) -> str:
        return """You are an expert ML experiment planner. Given a user's goal in natural language, you must produce a JSON experiment plan.

Analyse the goal to determine:
1. Problem type: "classification" or "regression"
2. Candidate models to try (pick 3-5 from: logistic_regression, random_forest, gradient_boosting, xgboost, svm, knn)
3. Evaluation metrics appropriate for the problem type
   - Classification: choose from f1, roc_auc, accuracy, precision, recall
   - Regression: choose from mse, rmse, r2, mae
4. Any hints about the target column name
5. A brief natural-language description of the approach

Respond ONLY with a valid JSON object in this exact format:
{
  "problem_type": "classification",
  "models": ["logistic_regression", "random_forest", "xgboost"],
  "metrics": ["f1", "roc_auc", "accuracy"],
  "primary_metric": "roc_auc",
  "target_column": "target",
  "approach": "Brief description of the planned approach"
}

Do NOT include any text outside the JSON object."""

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Create an experiment plan from the user goal.

        Args:
            state: Must contain ``goal`` (str).

        Returns:
            Partial state update with ``experiment_plan``.
        """
        goal = state["goal"]
        logger.info("[PlannerAgent] Planning for goal: %s", goal)

        prompt = f"User goal: {goal}"

        # If there's dataset info available, include it
        if state.get("dataset_summary"):
            prompt += (
                f"\n\nDataset summary is available:\n"
                f"Columns: {state['dataset_summary'].get('columns', [])}\n"
                f"Rows: {state['dataset_summary'].get('n_rows', 'unknown')}"
            )

        plan = self.invoke_llm_json(prompt)

        # Apply defaults if LLM omitted fields
        plan.setdefault("problem_type", "classification")
        plan.setdefault(
            "models", ["logistic_regression", "random_forest", "xgboost"]
        )
        plan.setdefault("metrics", ["f1", "roc_auc", "accuracy"])
        plan.setdefault("primary_metric", plan["metrics"][0])
        plan.setdefault("target_column", "target")
        plan.setdefault("approach", "Automated ML experiment")

        logger.info("[PlannerAgent] Plan created: %s", plan)
        return {"experiment_plan": plan}
