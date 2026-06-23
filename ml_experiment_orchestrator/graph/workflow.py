"""LangGraph Workflow — the stateful graph that orchestrates all agents.

Defines the full experiment lifecycle as a directed graph:

  planner → data_analysis → feature_engineering → runner → evaluation → critic
  critic ──[continue]──→ replanner → runner  (loop)
  critic ──[done]──────→ report → END
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from ml_experiment_orchestrator.agents.critic import CriticAgent
from ml_experiment_orchestrator.agents.data_analysis import DataAnalysisAgent
from ml_experiment_orchestrator.agents.evaluation import EvaluationAgent
from ml_experiment_orchestrator.agents.experiment_runner import ExperimentRunnerAgent
from ml_experiment_orchestrator.agents.feature_engineering import FeatureEngineeringAgent
from ml_experiment_orchestrator.agents.planner import PlannerAgent
from ml_experiment_orchestrator.agents.replanner import ReplannerAgent
from ml_experiment_orchestrator.agents.report import ReportAgent
from ml_experiment_orchestrator.graph.routing import should_continue_or_report
from ml_experiment_orchestrator.graph.state import ExperimentState

logger = logging.getLogger(__name__)

# ── Instantiate Agents (singletons for the graph) ─────────────────────────

_planner = PlannerAgent()
_data_analysis = DataAnalysisAgent()
_feature_engineering = FeatureEngineeringAgent()
_runner = ExperimentRunnerAgent()
_evaluation = EvaluationAgent()
_critic = CriticAgent()
_replanner = ReplannerAgent()
_report = ReportAgent()


# ── Node Functions ────────────────────────────────────────────────────────
# Each node receives the full state and returns a *partial* state update.


def planner_node(state: ExperimentState) -> dict[str, Any]:
    """Plan the experiment based on the user goal."""
    logger.info("═══ PLANNER NODE ═══")
    return _planner.run(state)


def data_analysis_node(state: ExperimentState) -> dict[str, Any]:
    """Profile and analyse the dataset."""
    logger.info("═══ DATA ANALYSIS NODE ═══")
    return _data_analysis.run(state)


def feature_engineering_node(state: ExperimentState) -> dict[str, Any]:
    """Design the preprocessing pipeline."""
    logger.info("═══ FEATURE ENGINEERING NODE ═══")
    return _feature_engineering.run(state)


def runner_node(state: ExperimentState) -> dict[str, Any]:
    """Train candidate models and log results."""
    logger.info("═══ EXPERIMENT RUNNER NODE ═══")
    return _runner.run(state)


def evaluation_node(state: ExperimentState) -> dict[str, Any]:
    """Evaluate and rank all models."""
    logger.info("═══ EVALUATION NODE ═══")
    return _evaluation.run(state)


def critic_node(state: ExperimentState) -> dict[str, Any]:
    """Critique results and decide whether to iterate."""
    logger.info("═══ CRITIC NODE ═══")
    return _critic.run(state)


def replanner_node(state: ExperimentState) -> dict[str, Any]:
    """Update the experiment plan based on critic feedback."""
    logger.info("═══ REPLANNER NODE ═══")
    return _replanner.run(state)


def report_node(state: ExperimentState) -> dict[str, Any]:
    """Generate the final report."""
    logger.info("═══ REPORT NODE ═══")
    return _report.run(state)


# ── Build the Graph ───────────────────────────────────────────────────────


def build_workflow() -> StateGraph:
    """Construct and compile the experiment orchestration graph.

    Returns:
        A compiled LangGraph ``StateGraph`` ready to invoke.
    """
    graph = StateGraph(ExperimentState)

    # Add nodes
    graph.add_node("planner", planner_node)
    graph.add_node("data_analysis", data_analysis_node)
    graph.add_node("feature_engineering", feature_engineering_node)
    graph.add_node("runner", runner_node)
    graph.add_node("evaluation", evaluation_node)
    graph.add_node("critic", critic_node)
    graph.add_node("replanner", replanner_node)
    graph.add_node("report", report_node)

    # Define edges (linear flow)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "data_analysis")
    graph.add_edge("data_analysis", "feature_engineering")
    graph.add_edge("feature_engineering", "runner")
    graph.add_edge("runner", "evaluation")
    graph.add_edge("evaluation", "critic")

    # Conditional routing from critic
    graph.add_conditional_edges(
        "critic",
        should_continue_or_report,
        {
            "replanner": "replanner",
            "report": "report",
        },
    )

    # Replanner loops back to runner
    graph.add_edge("replanner", "runner")

    # Report ends the workflow
    graph.add_edge("report", END)

    logger.info("Workflow graph built successfully")
    return graph.compile()


# Pre-built compiled workflow
workflow = build_workflow()
