"""Data Loader — dataset loading abstraction.

Supports:
  • Built-in sklearn demo datasets (heart_disease, diabetes, breast_cancer)
  • CSV files from the filesystem
  • Extensible interface for future connectors (e.g., MIMIC-III)
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
)

logger = logging.getLogger(__name__)

# ── Built-in Dataset Registry ────────────────────────────────────────────

_BUILTIN_DATASETS: dict[str, callable] = {
    "breast_cancer": load_breast_cancer,
    "diabetes": load_diabetes,
}


def _load_heart_disease() -> pd.DataFrame:
    """Load the UCI Heart Disease dataset from sklearn-compatible CSV.

    Since sklearn doesn't bundle heart disease directly, we construct it
    from the well-known feature set using the Statlog variant.
    """
    from sklearn.datasets import fetch_openml

    try:
        data = fetch_openml(name="heart-statlog", version=1, as_frame=True, parser="auto")
        df = data.frame  # type: ignore[union-attr]
        return df
    except Exception:
        logger.warning(
            "Could not fetch heart disease from OpenML. "
            "Falling back to breast_cancer dataset."
        )
        return load_builtin("breast_cancer")


def load_builtin(name: str) -> pd.DataFrame:
    """Load a built-in sklearn dataset as a pandas DataFrame.

    Args:
        name: One of 'heart_disease', 'diabetes', 'breast_cancer'.

    Returns:
        A DataFrame with features + target column.

    Raises:
        ValueError: If *name* is not a recognised built-in dataset.
    """
    if name == "heart_disease":
        return _load_heart_disease()

    loader = _BUILTIN_DATASETS.get(name)
    if loader is None:
        available = list(_BUILTIN_DATASETS.keys()) + ["heart_disease"]
        raise ValueError(
            f"Unknown built-in dataset '{name}'. Available: {available}"
        )

    data = loader()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    logger.info("Loaded built-in dataset '%s': %d×%d", name, *df.shape)
    return df


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file into a DataFrame.

    Args:
        path: Absolute or relative path to a CSV file.

    Returns:
        The loaded DataFrame.

    Raises:
        FileNotFoundError: If the path doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    logger.info("Loaded CSV '%s': %d×%d", path.name, *df.shape)
    return df


def load_dataset(identifier: str) -> pd.DataFrame:
    """Unified dataset loading: tries built-in names first, then CSV path.

    Args:
        identifier: A built-in dataset name or a path to a CSV file.

    Returns:
        The loaded DataFrame.
    """
    # Check if it's a built-in name
    builtins = list(_BUILTIN_DATASETS.keys()) + ["heart_disease"]
    if identifier in builtins:
        return load_builtin(identifier)

    # Otherwise treat as a file path
    return load_csv(identifier)
