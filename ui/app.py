"""Streamlit Dashboard — lightweight UI for the ML Experiment Orchestrator.

Provides:
  • Experiment creation form
  • Live status tracking
  • Results and metrics display
  • Final report viewer
"""

from __future__ import annotations

import time

import requests
import streamlit as st

# ── Configuration ─────────────────────────────────────────────────────────

API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="ML Experiment Orchestrator",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .status-badge {
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .status-pending { background: #fef3c7; color: #92400e; }
    .status-running { background: #dbeafe; color: #1e40af; }
    .status-completed { background: #d1fae5; color: #065f46; }
    .status-failed { background: #fee2e2; color: #991b1b; }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1e3a5f;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── API Helpers ───────────────────────────────────────────────────────────


def api_get(path: str) -> dict | list | None:
    """Make a GET request to the API."""
    try:
        resp = requests.get(f"{API_BASE}{path}", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to API. Is the server running?")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_post(path: str, data: dict) -> dict | None:
    """Make a POST request to the API."""
    try:
        resp = requests.post(f"{API_BASE}{path}", json=data, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to API. Is the server running?")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def status_badge(status: str) -> str:
    """Return an HTML badge for the experiment status."""
    return f'<span class="status-badge status-{status}">{status.upper()}</span>'


# ── Sidebar ───────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<p class="main-header">🧪 ML Orchestrator</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.subheader("🚀 New Experiment")

    goal = st.text_area(
        "Goal",
        placeholder="e.g., Predict heart disease from patient clinical data",
        height=80,
    )

    dataset = st.selectbox(
        "Dataset",
        ["heart_disease", "breast_cancer", "diabetes"],
        help="Select a built-in dataset or provide a CSV path",
    )

    if st.button("🎯 Launch Experiment", type="primary", use_container_width=True):
        if goal.strip():
            with st.spinner("Creating experiment..."):
                result = api_post("/experiment", {"goal": goal, "dataset": dataset})
                if result:
                    st.success(f"✅ Experiment created: `{result['id'][:8]}…`")
                    st.session_state["selected_experiment"] = result["id"]
                    time.sleep(1)
                    st.rerun()
        else:
            st.warning("Please enter a goal.")

    st.markdown("---")
    st.subheader("📋 Experiments")

    experiments = api_get("/experiments")
    if experiments and experiments.get("experiments"):
        for exp in experiments["experiments"]:
            label = f"{exp['goal'][:40]}… ({exp['status']})"
            if st.button(label, key=exp["id"], use_container_width=True):
                st.session_state["selected_experiment"] = exp["id"]
                st.rerun()
    else:
        st.info("No experiments yet. Create one above!")


# ── Main Content ──────────────────────────────────────────────────────────

st.markdown('<p class="main-header">ML Experiment Orchestrator</p>', unsafe_allow_html=True)
st.markdown(
    "An autonomous AI Research Scientist that plans, executes, evaluates, "
    "and improves machine learning experiments."
)
st.markdown("---")

selected_id = st.session_state.get("selected_experiment")

if selected_id:
    experiment = api_get(f"/experiment/{selected_id}")

    if experiment:
        # Header
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.subheader(experiment["goal"])
        with col2:
            st.markdown(status_badge(experiment["status"]), unsafe_allow_html=True)
        with col3:
            if experiment["status"] == "running":
                if st.button("🔄 Refresh"):
                    st.rerun()

        # Auto-refresh for running experiments
        if experiment["status"] == "running":
            st.info("⏳ Experiment is running... auto-refreshing in 10 seconds")
            time.sleep(10)
            st.rerun()

        # Tabs
        tab_overview, tab_runs, tab_report = st.tabs(
            ["📊 Overview", "🏃 Runs", "📝 Report"]
        )

        with tab_overview:
            # Best model metrics
            if experiment.get("best_model") and experiment["best_model"].get("metrics"):
                st.subheader("🏆 Best Model")
                best = experiment["best_model"]
                st.markdown(f"**Model:** `{best.get('model_name', 'N/A')}`")

                metrics = best.get("metrics", {})
                cols = st.columns(len(metrics))
                for col, (name, value) in zip(cols, metrics.items()):
                    with col:
                        st.markdown(
                            f"""<div class="metric-card">
                                <div class="metric-value">{value:.4f}</div>
                                <div class="metric-label">{name}</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )

            # Experiment plan
            if experiment.get("experiment_plan"):
                st.subheader("📋 Experiment Plan")
                plan = experiment["experiment_plan"]
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Problem Type:** `{plan.get('problem_type', 'N/A')}`")
                    st.markdown(f"**Models:** {', '.join(f'`{m}`' for m in plan.get('models', []))}")
                with col2:
                    st.markdown(f"**Primary Metric:** `{plan.get('primary_metric', 'N/A')}`")
                    st.markdown(f"**Metrics:** {', '.join(f'`{m}`' for m in plan.get('metrics', []))}")

            # Dataset summary
            if experiment.get("dataset_summary"):
                st.subheader("📊 Dataset Summary")
                ds = experiment["dataset_summary"]
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", ds.get("n_rows", "?"))
                with col2:
                    st.metric("Columns", ds.get("n_cols", "?"))
                with col3:
                    st.metric("Missing %", f"{ds.get('total_missing_pct', 0):.1f}%")

        with tab_runs:
            runs = experiment.get("runs", [])
            if runs:
                import pandas as pd

                rows = []
                for r in runs:
                    row = {
                        "Model": r["model_name"],
                        "Stage": r["stage"],
                        "Iteration": r["iteration"],
                    }
                    if r.get("metrics"):
                        row.update(
                            {k: round(v, 4) for k, v in r["metrics"].items()}
                        )
                    rows.append(row)

                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("No runs yet.")

        with tab_report:
            if experiment.get("final_report"):
                st.markdown(experiment["final_report"])
            elif experiment["status"] == "completed":
                report_data = api_get(f"/experiment/{selected_id}/report")
                if report_data and report_data.get("final_report"):
                    st.markdown(report_data["final_report"])
                else:
                    st.info("Report not available.")
            else:
                st.info("Report will be available after the experiment completes.")
    else:
        st.warning("Could not load experiment details.")
else:
    # Landing page
    st.markdown(
        """
        ### 🎯 How It Works

        1. **Describe your goal** — Tell the system what you want to predict
        2. **Select a dataset** — Choose from built-in datasets or provide your own
        3. **Launch** — The orchestrator autonomously:
           - 🎯 Plans the experiment
           - 📊 Analyses the dataset
           - 🔧 Designs preprocessing
           - 🏃 Trains multiple models
           - 📈 Evaluates results
           - 🔍 Critiques & improves
           - 📝 Generates a report

        ### 🤖 Multi-Agent Architecture

        The system uses **9 specialised AI agents** orchestrated via LangGraph:

        | Agent | Role |
        |-------|------|
        | 🎯 Planner | Understands goals, selects algorithms |
        | 📊 Data Analyst | Profiles datasets, finds issues |
        | 🔧 Feature Engineer | Designs preprocessing pipelines |
        | 🏃 Runner | Trains models, logs to MLflow |
        | ⚡ HPO | Optuna hyperparameter tuning |
        | 📈 Evaluator | Ranks models, computes metrics |
        | 🔍 Critic | Identifies improvements |
        | 🔄 Replanner | Generates improved plans |
        | 📝 Reporter | Creates comprehensive reports |

        ---
        *Create your first experiment using the sidebar →*
        """
    )
