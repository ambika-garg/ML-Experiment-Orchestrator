"""Replanner Agent — generates updated experiment plans from critic feedback.

Takes the Critic's improvement suggestions and translates them into concrete
changes to the experiment plan and feature engineering plan for the next
iteration.
"""

from __future__ import annotations

import logging
from typing import Any

from ml_experiment_orchestrator.agents.base import BaseAgent

logger = logging.getLogger(__name__)


class ReplannerAgent(BaseAgent):
    """Generate an updated experiment plan incorporating critic improvements."""

    @property
    def system_prompt(self) -> str:
        return """You are an expert ML experiment planner revising an experiment based on critic feedback.

Given:
1. The current experiment plan
2. The current feature engineering plan
3. Critic feedback with suggested improvements

Update both plans to incorporate the improvements. Be specific and actionable.

Available models: logistic_regression, random_forest, gradient_boosting, xgboost, svm, knn
Available scalers: standard, minmax, robust
Available encoders: onehot, label
Available imbalance methods: smote, random_oversampling, class_weight
Available feature selection: select_k_best, pca, variance_threshold

Respond ONLY with a valid JSON object:
{
  "experiment_plan": {
    "problem_type": "classification",
    "models": ["xgboost", "gradient_boosting", "svm"],
    "metrics": ["f1", "roc_auc", "accuracy"],
    "primary_metric": "roc_auc",
    "target_column": "target",
    "approach": "Updated approach description"
  },
  "feature_plan": {
    "preprocessing_steps": [
      {"step": "imputation", "method": "median", "columns": "numeric"},
      {"step": "scaling", "method": "robust", "columns": "numeric"},
      {"step": "encoding", "method": "onehot", "columns": "categorical"}
    ],
    "feature_selection": {"apply": true, "method": "select_k_best", "n_features": 15},
    "imbalance_handling": {"apply": true, "method": "smote"},
    "custom_code": "def transform_data(df: pd.DataFrame) -> pd.DataFrame:\n    # Update features based on critic suggestion or correct errors\n    return df",
    "reasoning": "Explanation of changes"
  }
}

Do NOT include any text outside the JSON object."""

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Generate updated plans from critic feedback.

        Args:
            state: Must contain ``critic_feedback``, ``experiment_plan``,
                   ``feature_plan``.

        Returns:
            Partial state with updated ``experiment_plan``, ``feature_plan``,
            and incremented ``iteration``.
        """
        feedback = state.get("critic_feedback", {})
        current_plan = state.get("experiment_plan", {})
        current_features = state.get("feature_plan", {})
        iteration = state.get("iteration", 0)

        logger.info(
            "[ReplannerAgent] Replanning for iteration %d with %d improvements",
            iteration + 1,
            len(feedback.get("improvements", [])),
        )

        improvements_text = "\n".join(
            f"  - {imp.get('action', '?')}: {imp.get('reason', '')}"
            for imp in feedback.get("improvements", [])
        )

        prompt = (
            f"Revise the experiment plan based on critic feedback.\n\n"
            f"Current experiment plan:\n{current_plan}\n\n"
            f"Current feature plan:\n{current_features}\n\n"
            f"Critic reasoning: {feedback.get('reasoning', 'N/A')}\n\n"
            f"Suggested improvements:\n{improvements_text}\n"
        )

        result = self.invoke_llm_json(prompt)

        # Extract updated plans, falling back to current ones
        new_plan = result.get("experiment_plan", current_plan)
        new_features = result.get("feature_plan", current_features)

        # Preserve essential fields from original plan
        new_plan.setdefault("problem_type", current_plan.get("problem_type", "classification"))
        new_plan.setdefault("target_column", current_plan.get("target_column", "target"))
        new_plan.setdefault("primary_metric", current_plan.get("primary_metric", "f1"))
        new_plan.setdefault("metrics", current_plan.get("metrics", ["f1"]))
        new_plan.setdefault("models", current_plan.get("models", ["random_forest"]))

        logger.info(
            "[ReplannerAgent] Updated plan: models=%s, imbalance=%s, feature_sel=%s",
            new_plan.get("models"),
            new_features.get("imbalance_handling", {}).get("apply"),
            new_features.get("feature_selection", {}).get("apply"),
        )

        return {
            "experiment_plan": new_plan,
            "feature_plan": new_features,
            "iteration": iteration + 1,
        }
