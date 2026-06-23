"""Report Agent — generates a comprehensive Markdown experiment report.

Produces a final document covering the entire experiment lifecycle:
executive summary, methodology, results, comparisons, and recommendations.
"""

from __future__ import annotations

import logging
from typing import Any

from ml_experiment_orchestrator.agents.base import BaseAgent

logger = logging.getLogger(__name__)


class ReportAgent(BaseAgent):
    """Generate a comprehensive Markdown report from the full experiment state."""

    @property
    def system_prompt(self) -> str:
        return """You are an expert ML research report writer. Given the full experiment context, generate a comprehensive, professional Markdown report.

The report MUST include these sections:
1. **Executive Summary** — Key findings in 2-3 sentences
2. **Objective** — The user's goal and problem type
3. **Dataset Overview** — Shape, features, quality, class balance
4. **Methodology** — Preprocessing pipeline, models tried, evaluation approach
5. **Results** — Detailed metrics table for ALL runs, organized by iteration
6. **Model Comparison** — Rankings with analysis of strengths/weaknesses
7. **Best Model** — Full details of the winner (name, parameters, all metrics)
8. **Improvement Journey** — How results improved across iterations (if multiple)
9. **Recommendations** — Next steps for further improvement
10. **Appendix** — Technical details, configuration

Use proper Markdown formatting with headers, tables, bold text, and bullet points.
Include a metrics comparison table with columns: Model | Stage | Iteration | Primary Metric | Other Metrics.

Respond with the full Markdown report as plain text (NOT wrapped in JSON)."""

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Generate the final experiment report.

        Args:
            state: The complete experiment state.

        Returns:
            Partial state with ``final_report`` (Markdown string).
        """
        logger.info("[ReportAgent] Generating final report")

        # Build comprehensive context
        plan = state.get("experiment_plan", {})
        summary = state.get("dataset_summary", {})
        feature_plan = state.get("feature_plan", {})
        results = state.get("experiment_results", [])
        evaluation = state.get("evaluation", {})
        best_model = state.get("best_model", {})
        critic_feedback = state.get("critic_feedback", {})

        # Build results table text
        results_text = "Run Results:\n"
        for r in results:
            if "error" not in r:
                metrics_str = ", ".join(
                    f"{k}={v:.4f}" for k, v in r.get("metrics", {}).items()
                )
                results_text += (
                    f"  - {r['model_name']} (iter={r.get('iteration', 0)}, "
                    f"stage={r.get('stage', 'initial')}): {metrics_str}\n"
                )

        prompt = (
            f"Generate a comprehensive ML experiment report.\n\n"
            f"User Goal: {state.get('goal', 'N/A')}\n\n"
            f"Experiment Plan:\n{plan}\n\n"
            f"Dataset Summary:\n"
            f"  Shape: {summary.get('n_rows', '?')} × {summary.get('n_cols', '?')}\n"
            f"  Numeric features: {len(summary.get('numeric_columns', []))}\n"
            f"  Categorical features: {len(summary.get('categorical_columns', []))}\n"
            f"  Missing data: {summary.get('total_missing_pct', 0)}%\n"
            f"  Class distribution: {summary.get('class_distribution', 'N/A')}\n\n"
            f"Feature Engineering:\n{feature_plan}\n\n"
            f"{results_text}\n"
            f"Best Model: {best_model.get('model_name', 'N/A')}\n"
            f"Best Metrics: {best_model.get('metrics', {})}\n"
            f"Best Parameters: {best_model.get('parameters', {})}\n\n"
            f"Rankings:\n"
        )

        for r in evaluation.get("rankings", []):
            prompt += f"  #{r['rank']} {r['model_name']} ({r.get('stage', '?')}): {r['primary_metric_value']:.4f}\n"

        prompt += (
            f"\nEvaluation Analysis: {evaluation.get('analysis', 'N/A')}\n"
            f"Critic Feedback: {critic_feedback.get('reasoning', 'N/A')}\n"
            f"Total Iterations: {state.get('iteration', 0) + 1}\n"
        )

        # Use plain text invoke (not JSON) for the report
        report = self.invoke_llm(prompt)

        # Clean up any leading/trailing artifacts
        report = report.strip()
        if report.startswith("```markdown"):
            report = report[len("```markdown") :].strip()
        if report.startswith("```"):
            report = report[3:].strip()
        if report.endswith("```"):
            report = report[:-3].strip()

        logger.info("[ReportAgent] Report generated (%d chars)", len(report))
        return {"final_report": report}
