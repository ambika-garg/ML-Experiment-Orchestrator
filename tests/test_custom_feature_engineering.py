from __future__ import annotations

import pandas as pd
import pytest
from ml_experiment_orchestrator.agents.experiment_runner import ExperimentRunnerAgent


def test_execute_custom_code_success() -> None:
    runner = ExperimentRunnerAgent()
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    custom_code = (
        "def transform_data(df: pd.DataFrame) -> pd.DataFrame:\n"
        "    df['new_col'] = df['col1'] + df['col2']\n"
        "    return df"
    )
    res = runner._execute_custom_code(df, custom_code)
    assert "new_col" in res.columns
    assert res["new_col"].tolist() == [5, 7, 9]


def test_execute_custom_code_missing_function() -> None:
    runner = ExperimentRunnerAgent()
    df = pd.DataFrame({"col1": [1, 2, 3]})
    custom_code = (
        "def transform(df):\n"
        "    return df"
    )
    with pytest.raises(ValueError, match="Function 'transform_data' not found in generated code"):
        runner._execute_custom_code(df, custom_code)


def test_apply_preprocessing_catches_error() -> None:
    runner = ExperimentRunnerAgent()
    df = pd.DataFrame({"col1": [1, 2, 3]})
    feature_plan = {
        "custom_code": (
            "def transform_data(df: pd.DataFrame) -> pd.DataFrame:\n"
            "    raise ValueError('Custom error occurred!')"
        ),
        "preprocessing_steps": [],
    }
    # _apply_preprocessing splits X and y, applies steps. Let's call it.
    X = df.copy()
    y = pd.Series([1, 0, 1])
    X_out, y_out = runner._apply_preprocessing(X, y, feature_plan, "classification")

    # The error should be in feature_plan["error"]
    assert "error" in feature_plan
    assert "ValueError: Custom error occurred!" in feature_plan["error"]
    # The pipeline should still execute successfully (falling back to baseline shapes)
    assert X_out.shape == (3, 1)
