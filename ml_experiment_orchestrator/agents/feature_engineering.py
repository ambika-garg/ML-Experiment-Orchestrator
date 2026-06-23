"""Feature Engineering Agent — plans preprocessing steps based on data characteristics.

Uses LLM reasoning over the dataset summary to decide which transformations
to apply: scaling, encoding, imputation, feature selection, and imbalance handling.
"""

from __future__ import annotations

import logging
from typing import Any

from ml_experiment_orchestrator.agents.base import BaseAgent

logger = logging.getLogger(__name__)


class FeatureEngineeringAgent(BaseAgent):
    """Determine the optimal preprocessing pipeline for the dataset."""

    @property
    def system_prompt(self) -> str:
        return """You are an expert ML feature engineer. Given a dataset summary and experiment plan, you must design an optimal preprocessing pipeline AND generate custom feature engineering Python code.

Consider:
1. Scaling: StandardScaler, MinMaxScaler, RobustScaler (for numeric features)
2. Encoding: OneHotEncoder, LabelEncoder, OrdinalEncoder (for categorical features)
3. Imputation: mean, median, most_frequent, knn (for missing values)
4. Feature selection: SelectKBest, PCA, variance_threshold
5. Imbalance handling: smote, random_oversampling, class_weight (for imbalanced classification)

Your custom feature engineering code must define a function `transform_data(df: pd.DataFrame) -> pd.DataFrame` that creates new informative features based on the columns description.

Rules for writing `transform_data`:
- pandas and numpy are already imported in context as `pd` and `np`.
- Never delete the target column (it is handled separately and will not be present in `df`).
- Handle divisions by zero or NaN values safely (e.g. use `.fillna(0)` or add small epsilon values to denominators like `1e-6`).
- Return a DataFrame with the new features appended. Keep original features unless they are completely redundant.
- Ensure the code is syntactically valid.

Respond ONLY with a valid JSON object in this format:
{
  "preprocessing_steps": [
    {"step": "imputation", "method": "median", "columns": "numeric"},
    {"step": "imputation", "method": "most_frequent", "columns": "categorical"},
    {"step": "scaling", "method": "standard", "columns": "numeric"},
    {"step": "encoding", "method": "onehot", "columns": "categorical"}
  ],
  "feature_selection": {
    "apply": false,
    "method": null,
    "n_features": null
  },
  "imbalance_handling": {
    "apply": false,
    "method": null
  },
  "custom_code": "def transform_data(df: pd.DataFrame) -> pd.DataFrame:\n    # Create feature ratios or segmentations safely\n    # df['ratio_col'] = df['col1'] / (df['col2'] + 1e-6)\n    return df",
  "reasoning": "Brief explanation of choices"
}

Do NOT include any text outside the JSON object."""

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Design a feature engineering plan.

        Args:
            state: Must contain ``dataset_summary`` and ``experiment_plan``.

        Returns:
            Partial state update with ``feature_plan``.
        """
        summary = state["dataset_summary"]
        plan = state["experiment_plan"]

        logger.info("[FeatureEngineeringAgent] Designing preprocessing pipeline")

        prompt = (
            f"Design a preprocessing pipeline for this dataset.\n\n"
            f"Problem type: {plan.get('problem_type', 'classification')}\n"
            f"Target column: {plan.get('target_column', 'target')}\n\n"
            f"Dataset summary:\n"
            f"  Shape: {summary['n_rows']} × {summary['n_cols']}\n"
            f"  Numeric columns: {summary.get('numeric_columns', [])}\n"
            f"  Categorical columns: {summary.get('categorical_columns', [])}\n"
            f"  Missing values: {summary.get('missing_values', {})}\n"
            f"  Class distribution: {summary.get('class_distribution', 'N/A')}\n"
            f"  Total missing %: {summary.get('total_missing_pct', 0)}\n"
        )

        if summary.get("llm_analysis", {}).get("recommendations"):
            prompt += (
                f"\nData analysis recommendations:\n"
                + "\n".join(
                    f"  - {r}" for r in summary["llm_analysis"]["recommendations"]
                )
            )

        feature_plan = self.invoke_llm_json(prompt)

        # Apply defaults
        feature_plan.setdefault("preprocessing_steps", [
            {"step": "scaling", "method": "standard", "columns": "numeric"},
        ])
        feature_plan.setdefault(
            "feature_selection", {"apply": False, "method": None, "n_features": None}
        )
        feature_plan.setdefault(
            "imbalance_handling", {"apply": False, "method": None}
        )

        logger.info(
            "[FeatureEngineeringAgent] Plan: %d steps, feature_selection=%s, imbalance=%s",
            len(feature_plan.get("preprocessing_steps", [])),
            feature_plan["feature_selection"].get("apply"),
            feature_plan["imbalance_handling"].get("apply"),
        )
        return {"feature_plan": feature_plan}
