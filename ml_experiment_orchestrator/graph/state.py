"""ExperimentState — the shared state object flowing through the LangGraph workflow.

Uses TypedDict with Annotated reducers so that list fields (like experiment_results)
accumulate across nodes rather than being overwritten.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class ExperimentState(TypedDict, total=False):
    """State that flows through every node in the experiment graph.

    Fields marked with ``Annotated[..., operator.add]`` use an append-reducer:
    each node returns a *list of new items* which are concatenated onto the
    existing list rather than replacing it.
    """

    # ── User Input ────────────────────────────────────────────────────────
    goal: str
    dataset_path: str

    # ── Data (not serialized to DB — lives in memory during graph execution)
    dataset: Any  # pd.DataFrame

    # ── Agent Outputs ─────────────────────────────────────────────────────
    experiment_plan: dict
    dataset_summary: dict
    feature_plan: dict
    experiment_results: Annotated[list, operator.add]  # Append across iterations
    evaluation: dict
    critic_feedback: dict
    best_model: dict
    final_report: str

    # ── Control Flow ──────────────────────────────────────────────────────
    iteration: int
    max_iterations: int
    experiment_id: str  # Database experiment UUID

    # ── Internal (shared between runner / HPO) ────────────────────────────
    _X_train: Any  # np.ndarray
    _y_train: Any
    _X_test: Any
    _y_test: Any
