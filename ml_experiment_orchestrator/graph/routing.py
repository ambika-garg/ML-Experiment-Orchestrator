"""Conditional routing functions for the LangGraph workflow.

These are used as edge conditions to determine which node runs next.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def should_continue_or_report(state: dict[str, Any]) -> str:
    """After the Critic node: decide whether to replan, run HPO, or generate report.

    Returns:
        ``"hpo"`` if hyperparameter tuning is requested,
        ``"replanner"`` if other improvements are needed,
        ``"report"`` otherwise.
    """
    feedback = state.get("critic_feedback", {})
    should_continue = feedback.get("should_continue", False)

    if should_continue:
        improvements = feedback.get("improvements", [])
        has_hpo = any(imp.get("action") == "tune_hyperparameters" for imp in improvements)
        if has_hpo:
            logger.info("[Routing] Critic says continue → hpo")
            return "hpo"
        logger.info("[Routing] Critic says continue → replanner")
        return "replanner"
    else:
        logger.info("[Routing] Critic says stop → report")
        return "report"
