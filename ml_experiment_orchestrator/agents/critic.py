"""Critic Agent — analyses experiment results and determines whether to improve.

The Critic reviews evaluation results, identifies weaknesses, and either:
  • Proposes concrete improvements (continue iterating), or
  • Declares convergence (stop and generate report)
"""

from __future__ import annotations

import logging
from typing import Any

from ml_experiment_orchestrator.agents.base import BaseAgent
from ml_experiment_orchestrator.config import settings

logger = logging.getLogger(__name__)


class CriticAgent(BaseAgent):
    """Critique experiment results and decide whether to iterate."""

    @property
    def system_prompt(self) -> str:
        return """You are an expert ML research critic. Your job is to analyse experiment results and determine if improvements are possible.

Consider:
1. Are the metrics satisfactory for the problem type?
2. Is there significant room for improvement?
3. Have we tried enough model types and preprocessing approaches?
4. Are there signs of overfitting or underfitting?
5. Could additional feature engineering help?
6. Would hyperparameter tuning (if not yet done) improve results?

Available improvement actions:
- "apply_smote": Apply SMOTE for class imbalance
- "add_pca": Apply PCA for dimensionality reduction
- "try_model:<model_name>": Try a new model (e.g., "try_model:svm")
- "tune_hyperparameters": Run Optuna HPO on the best model
- "add_feature_selection": Apply SelectKBest feature selection
- "change_scaler:robust": Switch to a different scaler
- "increase_cv_folds": Use more cross-validation folds

Respond ONLY with a valid JSON object:
{
  "should_continue": true,
  "improvements": [
    {"action": "apply_smote", "reason": "Class imbalance detected"},
    {"action": "try_model:gradient_boosting", "reason": "Ensemble methods might perform better"}
  ],
  "reasoning": "Detailed explanation of your critique",
  "confidence": 0.7
}

Set "should_continue" to false if:
- Metrics are already very good (>0.95 for classification)
- Previous iteration showed <1% improvement
- All reasonable approaches have been tried

Do NOT include any text outside the JSON object."""

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Critique the current results and decide whether to continue.

        Args:
            state: Must contain ``evaluation``, ``experiment_results``,
                   ``experiment_plan``, ``dataset_summary``.

        Returns:
            Partial state with ``critic_feedback``.
        """
        evaluation = state.get("evaluation", {})
        results = state.get("experiment_results", [])
        plan = state.get("experiment_plan", {})
        summary = state.get("dataset_summary", {})
        iteration = state.get("iteration", 0)
        max_iterations = state.get("max_iterations", settings.max_iterations)
        feature_plan = state.get("feature_plan", {})

        primary_metric = plan.get("primary_metric", "f1")
        best_value = evaluation.get("primary_metric_value", 0)

        logger.info(
            "[CriticAgent] Reviewing iteration %d — best %s=%.4f",
            iteration,
            primary_metric,
            best_value,
        )

        # ── Check Automatic Termination ───────────────────────────────────
        if iteration >= max_iterations:
            logger.info("[CriticAgent] Max iterations (%d) reached", max_iterations)
            return {
                "critic_feedback": {
                    "should_continue": False,
                    "improvements": [],
                    "reasoning": f"Maximum iterations ({max_iterations}) reached.",
                    "confidence": 1.0,
                }
            }

        # Check for convergence (improvement < threshold)
        if iteration > 0:
            prev_results = [r for r in results if r.get("iteration", 0) < iteration]
            if prev_results:
                prev_best = max(
                    prev_results,
                    key=lambda r: r.get("metrics", {}).get(primary_metric, 0),
                )
                prev_value = prev_best.get("metrics", {}).get(primary_metric, 0)
                improvement = abs(best_value - prev_value)
                if improvement < settings.convergence_threshold:
                    logger.info(
                        "[CriticAgent] Converged — improvement %.4f < threshold %.4f",
                        improvement,
                        settings.convergence_threshold,
                    )
                    return {
                        "critic_feedback": {
                            "should_continue": False,
                            "improvements": [],
                            "reasoning": (
                                f"Converged: improvement ({improvement:.4f}) is below "
                                f"threshold ({settings.convergence_threshold})."
                            ),
                            "confidence": 1.0,
                        }
                    }

        # ── LLM Critique ─────────────────────────────────────────────────
        # Build context for the LLM
        rankings_text = ""
        for r in evaluation.get("rankings", [])[:10]:
            rankings_text += (
                f"  #{r['rank']} {r['model_name']} ({r.get('stage', '?')}): "
                f"{r['primary_metric_value']:.4f}\n"
            )

        prompt = (
            f"Review these ML experiment results and suggest improvements.\n\n"
            f"Goal: {state.get('goal', 'N/A')}\n"
            f"Problem type: {plan.get('problem_type', 'classification')}\n"
            f"Primary metric: {primary_metric} = {best_value:.4f}\n"
            f"Iteration: {iteration}/{max_iterations}\n\n"
            f"Rankings:\n{rankings_text}\n"
            f"Dataset: {summary.get('n_rows', '?')} rows × {summary.get('n_cols', '?')} cols\n"
            f"Class distribution: {summary.get('class_distribution', 'N/A')}\n"
            f"Missing data: {summary.get('total_missing_pct', 0)}%\n\n"
            f"Current preprocessing:\n"
            f"  Imbalance handling: {feature_plan.get('imbalance_handling', {})}\n"
            f"  Feature selection: {feature_plan.get('feature_selection', {})}\n\n"
            f"Previous analysis concerns: {evaluation.get('concerns', [])}\n"
        )

        feedback = self.invoke_llm_json(prompt)
        feedback.setdefault("should_continue", False)
        feedback.setdefault("improvements", [])
        feedback.setdefault("reasoning", "No specific reasoning provided")

        logger.info(
            "[CriticAgent] should_continue=%s, %d improvements proposed",
            feedback["should_continue"],
            len(feedback["improvements"]),
        )
        return {"critic_feedback": feedback}
