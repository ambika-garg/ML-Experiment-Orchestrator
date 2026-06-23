"""Data Analysis Agent — profiles the dataset and produces actionable insights.

Combines programmatic pandas analysis with LLM interpretation to produce
a comprehensive dataset summary including feature types, missing values,
class balance, and preprocessing recommendations.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from ml_experiment_orchestrator.agents.base import BaseAgent

logger = logging.getLogger(__name__)


class DataAnalysisAgent(BaseAgent):
    """Profile a DataFrame and return a structured dataset summary."""

    @property
    def system_prompt(self) -> str:
        return """You are an expert data scientist specialising in dataset analysis. Given a statistical summary of a dataset, you must produce a JSON analysis report.

            Analyse the data profile to identify:
            1. Data quality issues (missing values, outliers)
            2. Feature types and their suitability
            3. Class balance (for classification)
            4. Correlations and potential redundancies
            5. Specific preprocessing recommendations

            Respond ONLY with a valid JSON object in this format:
            {
            "quality_assessment": "Brief assessment of data quality",
            "recommendations": [
                "Specific recommendation 1",
                "Specific recommendation 2"
            ],
            "potential_issues": [
                "Issue 1"
            ],
            "feature_importance_hints": ["col1", "col2"]
            }

            Do NOT include any text outside the JSON object."""

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Analyse the dataset and return a structured summary.

        Args:
            state: Must contain ``dataset`` (pd.DataFrame) and ``experiment_plan``.

        Returns:
            Partial state update with ``dataset_summary``.
        """
        df: pd.DataFrame = state["dataset"]
        plan: dict = state.get("experiment_plan", {})
        target_col = plan.get("target_column", "target")

        logger.info("[DataAnalysisAgent] Profiling dataset (%d×%d)", *df.shape)

        # ── Programmatic Profiling ────────────────────────────────────────
        summary = self._profile(df, target_col)

        # ── LLM Interpretation ────────────────────────────────────────────
        profile_text = self._format_profile_for_llm(summary)
        prompt = (
            f"Analyse this dataset profile and provide recommendations.\n\n"
            f"Goal: {state.get('goal', 'N/A')}\n"
            f"Problem type: {plan.get('problem_type', 'unknown')}\n\n"
            f"{profile_text}"
        )
        llm_analysis = self.invoke_llm_json(prompt)
        summary["llm_analysis"] = llm_analysis

        logger.info("[DataAnalysisAgent] Summary complete")
        return {"dataset_summary": summary}

    # ── Private Helpers ───────────────────────────────────────────────────

    def _profile(self, df: pd.DataFrame, target_col: str) -> dict[str, Any]:
        """Build a comprehensive statistical profile of the dataframe."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        # Missing values
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        missing_info = {
            col: {"count": int(missing[col]), "percent": float(missing_pct[col])}
            for col in df.columns
            if missing[col] > 0
        }

        # Class distribution (for classification)
        class_distribution = None
        if target_col in df.columns:
            vc = df[target_col].value_counts()
            class_distribution = {
                str(k): int(v) for k, v in vc.items()
            }

        # Basic stats for numeric columns
        stats = {}
        if numeric_cols:
            desc = df[numeric_cols].describe().to_dict()
            stats = {
                col: {k: round(float(v), 4) for k, v in vals.items()}
                for col, vals in desc.items()
            }

        # Correlations (top pairs)
        correlations = {}
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            # Get top 10 absolute correlations (excluding self)
            pairs = []
            for i, col_a in enumerate(numeric_cols):
                for col_b in numeric_cols[i + 1 :]:
                    pairs.append(
                        (col_a, col_b, abs(corr_matrix.loc[col_a, col_b]))
                    )
            pairs.sort(key=lambda x: x[2], reverse=True)
            correlations = {
                f"{a} ↔ {b}": round(float(c), 4) for a, b, c in pairs[:10]
            }

        return {
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "columns": df.columns.tolist(),
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "feature_types": {
                col: str(df[col].dtype) for col in df.columns
            },
            "missing_values": missing_info,
            "total_missing_pct": round(
                float(df.isnull().sum().sum() / df.size * 100), 2
            ),
            "class_distribution": class_distribution,
            "statistics": stats,
            "top_correlations": correlations,
            "target_column": target_col,
        }

    @staticmethod
    def _format_profile_for_llm(summary: dict) -> str:
        """Format the profile dict into a readable string for the LLM."""
        lines = [
            f"Shape: {summary['n_rows']} rows × {summary['n_cols']} columns",
            f"Numeric features ({len(summary['numeric_columns'])}): {summary['numeric_columns'][:15]}",
            f"Categorical features ({len(summary['categorical_columns'])}): {summary['categorical_columns'][:15]}",
            f"Missing values: {summary['total_missing_pct']}% overall",
        ]
        if summary.get("missing_values"):
            lines.append(
                f"  Columns with missing: {list(summary['missing_values'].keys())[:10]}"
            )
        if summary.get("class_distribution"):
            lines.append(f"Class distribution: {summary['class_distribution']}")
        if summary.get("top_correlations"):
            top3 = dict(list(summary["top_correlations"].items())[:3])
            lines.append(f"Top correlations: {top3}")
        return "\n".join(lines)
