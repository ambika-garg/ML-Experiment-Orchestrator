"""Conditional routing functions for the LangGraph workflow.

These are used as edge conditions to determine which node runs next.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def should_continue_or_report(state: dict[str, Any]) -> str:
    """After the Critic node: decide whether to replan or generate report.

    Returns:
        ``"replanner"`` if improvements are needed, ``"report"`` otherwise.
    """
    feedback = state.get("critic_feedback", {})
    should_continue = feedback.get("should_continue", False)

    if should_continue:
        logger.info("[Routing] Critic says continue → replanner")
        return "replanner"
    else:
        logger.info("[Routing] Critic says stop → report")
        return "report"
