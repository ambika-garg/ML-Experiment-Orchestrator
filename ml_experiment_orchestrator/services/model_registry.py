"""Model Registry — maps model names to sklearn/xgboost classes and hyperparameter search spaces.

Provides:
  • Model instantiation by name
  • Default hyperparameter ranges for Optuna tuning
  • Problem-type-aware model selection (classifier vs regressor)
"""

from __future__ import annotations

import logging
from typing import Any

from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

logger = logging.getLogger(__name__)


# ── Registry ──────────────────────────────────────────────────────────────

_CLASSIFIERS: dict[str, type] = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "xgboost": XGBClassifier,
    "svm": SVC,
    "knn": KNeighborsClassifier,
}

_REGRESSORS: dict[str, type] = {
    "logistic_regression": Ridge,  # substitute for regression
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
    "xgboost": XGBRegressor,
    "svm": SVR,
    "knn": KNeighborsRegressor,
}


def get_model(
    name: str,
    problem_type: str = "classification",
    params: dict[str, Any] | None = None,
) -> Any:
    """Instantiate a model by name with optional parameters.

    Args:
        name: Model name (e.g. ``'xgboost'``).
        problem_type: ``'classification'`` or ``'regression'``.
        params: Optional keyword arguments passed to the constructor.

    Returns:
        An instantiated sklearn-compatible estimator.

    Raises:
        ValueError: If the model name is not recognised.
    """
    registry = _CLASSIFIERS if problem_type == "classification" else _REGRESSORS
    cls = registry.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown model '{name}' for {problem_type}. "
            f"Available: {list(registry.keys())}"
        )
    params = params or {}

    # Defaults for specific models
    if name == "xgboost":
        params.setdefault("eval_metric", "logloss" if problem_type == "classification" else "rmse")
        params.setdefault("verbosity", 0)
        params.setdefault("use_label_encoder", False)
    if name == "svm":
        params.setdefault("probability", True)
    if name == "logistic_regression" and problem_type == "classification":
        params.setdefault("max_iter", 1000)
        params.setdefault("solver", "lbfgs")

    model = cls(**params)
    logger.debug("Created %s(%s) for %s", name, params, problem_type)
    return model


def get_available_models(problem_type: str = "classification") -> list[str]:
    """Return list of available model names for the given problem type."""
    registry = _CLASSIFIERS if problem_type == "classification" else _REGRESSORS
    return list(registry.keys())


# ── Optuna Search Spaces ─────────────────────────────────────────────────

def get_search_space(name: str, trial: Any) -> dict[str, Any]:
    """Return a hyperparameter dict using Optuna trial suggestions.

    Args:
        name: Model name.
        trial: An ``optuna.Trial`` instance.

    Returns:
        Dict of suggested hyperparameters.
    """
    if name == "logistic_regression":
        return {
            "C": trial.suggest_float("C", 0.001, 100, log=True),
            "max_iter": 1000,
            "solver": "lbfgs",
        }

    if name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2"]
            ),
        }

    if name == "gradient_boosting":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        }

    if name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "verbosity": 0,
            "use_label_encoder": False,
        }

    if name == "svm":
        return {
            "C": trial.suggest_float("C", 0.01, 100, log=True),
            "kernel": trial.suggest_categorical("kernel", ["rbf", "poly"]),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
            "probability": True,
        }

    if name == "knn":
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 30),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "metric": trial.suggest_categorical(
                "metric", ["euclidean", "manhattan", "minkowski"]
            ),
        }

    # Fallback: no tunable params
    return {}
