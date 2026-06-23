"""Streamlit Dashboard — lightweight UI for the ML Experiment Orchestrator.

Provides:
  • Experiment creation form
  • Live status tracking
  • Results and metrics display
  • Final report viewer
"""

from __future__ import annotations

import time
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

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
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"], [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
    }

    /* Main Container Styles */
    .stApp {
        background-color: #0b0f19;
        color: #f3f4f6;
    }

    /* Sidebar Custom Styling */
    section[data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid #1f2937;
    }

    /* Main Header Styling with Radial Gradient */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        letter-spacing: -0.025em;
    }

    /* Status Badges */
    .status-badge {
        padding: 6px 16px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        display: inline-block;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    }
    .status-pending { 
        background: rgba(245, 158, 11, 0.15); 
        color: #fbbf24; 
        border: 1px solid rgba(245, 158, 11, 0.3); 
    }
    .status-running { 
        background: rgba(59, 130, 246, 0.15); 
        color: #60a5fa; 
        border: 1px solid rgba(59, 130, 246, 0.3); 
    }
    .status-completed { 
        background: rgba(16, 185, 129, 0.15); 
        color: #34d399; 
        border: 1px solid rgba(16, 185, 129, 0.3); 
    }
    .status-failed { 
        background: rgba(239, 68, 68, 0.15); 
        color: #f87171; 
        border: 1px solid rgba(239, 68, 68, 0.3); 
    }

    /* Glassmorphic Metric Cards */
    .metric-card {
        background: rgba(17, 24, 39, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 1.25rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(99, 102, 241, 0.4);
    }
    .metric-value {
        font-size: 2.25rem;
        font-weight: 700;
        background: linear-gradient(135deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        margin-top: 0.25rem;
    }

    /* Landing Page Grid Cards */
    .agent-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1.5rem;
        margin-top: 1.5rem;
        margin-bottom: 2rem;
    }
    .agent-card {
        background: rgba(17, 24, 39, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 1.5rem;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    .agent-card:hover {
        transform: translateY(-5px);
        border-color: rgba(139, 92, 246, 0.3);
        box-shadow: 0 10px 30px rgba(139, 92, 246, 0.1);
    }
    .agent-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.25rem;
        font-weight: 700;
        color: #f3f4f6;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .agent-role {
        font-size: 0.9rem;
        color: #9ca3af;
        line-height: 1.5;
    }

    /* Tabs Styling Overrides */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(17, 24, 39, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px 8px 0px 0px;
        padding: 8px 16px;
        color: #9ca3af;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(99, 102, 241, 0.15) !important;
        border-color: rgba(99, 102, 241, 0.4) !important;
        color: #6366f1 !important;
    }

    /* Tables & Dataframes styling override */
    div[data-testid="stTable"] table {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.05);
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


def render_workflow_svg(status: str, runs: list) -> str:
    """Return a styled SVG flow chart showing active/completed agent nodes."""
    steps = [
        {"id": "planner", "label": "Planner", "x": 50, "y": 50, "icon": "🎯"},
        {"id": "analyst", "label": "Data Analyst", "x": 140, "y": 50, "icon": "📊"},
        {"id": "feat_eng", "label": "Feature Eng", "x": 230, "y": 50, "icon": "🔧"},
        {"id": "runner", "label": "Runner", "x": 320, "y": 50, "icon": "🏃"},
        {"id": "eval", "label": "Evaluator", "x": 410, "y": 50, "icon": "📈"},
        {"id": "critic", "label": "Critic", "x": 500, "y": 50, "icon": "🔍"},
        {"id": "hpo", "label": "HPO", "x": 600, "y": 25, "icon": "⚡"},
        {"id": "replan", "label": "Replanner", "x": 600, "y": 75, "icon": "🔄"},
        {"id": "report", "label": "Reporter", "x": 710, "y": 50, "icon": "📝"},
    ]

    active_node = ""
    completed_nodes = set()

    if status == "pending":
        active_node = "planner"
    elif status == "completed":
        completed_nodes = {"planner", "analyst", "feat_eng", "runner", "eval", "critic", "hpo", "replan", "report"}
    elif status == "failed":
        completed_nodes = {"planner", "analyst", "feat_eng"}
        active_node = "runner"
    elif status == "running":
        if not runs:
            active_node = "planner"
        else:
            stages = {r.get("stage") for r in runs}
            has_hpo = "hpo" in stages
            has_replan = "replan" in stages or any(r.get("iteration", 0) > 0 for r in runs)
            
            completed_nodes.add("planner")
            completed_nodes.add("analyst")
            completed_nodes.add("feat_eng")
            
            if has_hpo:
                completed_nodes.update({"runner", "eval", "critic"})
                active_node = "hpo"
            elif has_replan:
                completed_nodes.update({"runner", "eval", "critic"})
                active_node = "replan"
            else:
                completed_nodes.add("runner")
                active_node = "eval"

    lines_html = ""
    def get_line(x1, y1, x2, y2, active):
        stroke = "#818cf8" if active else "#374151"
        glow = "filter='url(#glow)'" if active else ""
        return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="2" {glow} />'

    # Draw lines
    lines_html += get_line(50, 50, 140, 50, "analyst" in completed_nodes or active_node == "analyst")
    lines_html += get_line(140, 50, 230, 50, "feat_eng" in completed_nodes or active_node == "feat_eng")
    lines_html += get_line(230, 50, 320, 50, "runner" in completed_nodes or active_node == "runner")
    lines_html += get_line(320, 50, 410, 50, "eval" in completed_nodes or active_node == "eval")
    lines_html += get_line(410, 50, 500, 50, "critic" in completed_nodes or active_node == "critic")

    # Critic to HPO / Replanner
    lines_html += get_line(500, 50, 600, 25, "hpo" in completed_nodes or active_node == "hpo")
    lines_html += get_line(500, 50, 600, 75, "replan" in completed_nodes or active_node == "replan")

    # HPO / Replanner back loops (dotted/dashed)
    stroke_hpo = "#818cf8" if "hpo" in completed_nodes else "#374151"
    lines_html += f'<path d="M 600,25 Q 505,0 410,34" fill="none" stroke="{stroke_hpo}" stroke-width="1.5" stroke-dasharray="3,3" />'

    stroke_replan = "#818cf8" if "replan" in completed_nodes else "#374151"
    lines_html += f'<path d="M 600,75 Q 460,95 320,66" fill="none" stroke="{stroke_replan}" stroke-width="1.5" stroke-dasharray="3,3" />'

    # HPO / Replanner / Critic to Reporter
    lines_html += get_line(500, 50, 710, 50, "report" in completed_nodes or active_node == "report")
    if "hpo" in completed_nodes:
        lines_html += get_line(600, 25, 710, 50, "report" in completed_nodes or active_node == "report")
    if "replan" in completed_nodes:
        lines_html += get_line(600, 75, 710, 50, "report" in completed_nodes or active_node == "report")

    nodes_html = ""
    for step in steps:
        nid = step["id"]
        is_active = (nid == active_node)
        is_completed = nid in completed_nodes

        color = "#10b981" if is_completed else ("#6366f1" if is_active else "#1f2937")
        stroke = "#34d399" if is_completed else ("#818cf8" if is_active else "#4b5563")
        text_color = "#ffffff" if (is_active or is_completed) else "#9ca3af"
        radius = 16 if is_active else 12
        glow = "filter='url(#glow)'" if is_active else ""

        nodes_html += f"""
        <g transform="translate({step['x']}, {step['y']})">
            <circle cx="0" cy="0" r="{radius}" fill="{color}" stroke="{stroke}" stroke-width="2" {glow} />
            <text x="0" y="4" font-size="12" text-anchor="middle">{step['icon']}</text>
            <text x="0" y="24" font-size="10" font-family="'Outfit', sans-serif" fill="{text_color}" text-anchor="middle" font-weight="600">{step['label']}</text>
        </g>
        """

    svg = f"""
    <svg viewBox="0 0 760 110" width="100%" xmlns="http://www.w3.org/2000/svg" style="margin-bottom: 20px;">
        <defs>
            <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
                <feGaussianBlur stdDeviation="4" result="blur" />
                <feMerge>
                    <feMergeNode in="blur" />
                    <feMergeNode in="SourceGraphic" />
                </feMerge>
            </filter>
        </defs>
        <rect width="100%" height="100%" fill="rgba(17, 24, 39, 0.4)" rx="16" />
        {lines_html}
        {nodes_html}
    </svg>
    """
    return svg


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

        # Multi-Agent progress visualization
        svg_code = render_workflow_svg(experiment["status"], experiment.get("runs", []))
        st.markdown(svg_code, unsafe_allow_html=True)
        st.markdown("---")

        # Auto-refresh for running experiments
        if experiment["status"] == "running":
            st.info("⏳ Experiment is running... auto-refreshing in 10 seconds")
            time.sleep(10)
            st.rerun()

        # Tabs
        tab_overview, tab_runs, tab_report, tab_llm = st.tabs(
            ["📊 Overview", "🏃 Runs", "📝 Report", "🤖 LLM Observability"]
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

                df_runs = pd.DataFrame(rows)
                
                st.subheader("📋 Experimental Trials & Runs")
                st.dataframe(df_runs, use_container_width=True, hide_index=True)
                
                # Visual performance comparison
                st.markdown("---")
                st.subheader("📈 Performance Comparison")
                
                primary_metric = "f1"
                if experiment.get("experiment_plan"):
                    primary_metric = experiment["experiment_plan"].get("primary_metric", "f1")
                
                # Ensure the primary metric is in the dataframe
                available_metrics = [col for col in df_runs.columns if col not in ["Model", "Stage", "Iteration"]]
                if primary_metric not in available_metrics and available_metrics:
                    primary_metric = available_metrics[0]
                    
                if primary_metric in df_runs.columns:
                    df_plot = df_runs.copy()
                    df_plot["Stage"] = df_plot["Stage"].map({
                        "initial": "Baseline Stage",
                        "hpo": "HPO Tuning",
                        "replan": "Replan Improvement"
                    }).fillna(df_plot["Stage"])
                    
                    is_lower_better = primary_metric.lower() in ["rmse", "mae", "mse"]
                    direction_text = "(Lower is Better)" if is_lower_better else "(Higher is Better)"
                    
                    fig = px.bar(
                        df_plot,
                        x="Model",
                        y=primary_metric,
                        color="Stage",
                        barmode="group",
                        title=f"Model Comparison by {primary_metric.upper()} {direction_text}",
                        color_discrete_map={
                            "Baseline Stage": "#6366f1",
                            "HPO Tuning": "#10b981",
                            "Replan Improvement": "#a855f7"
                        },
                        text_auto=".4f"
                    )
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Inter, sans-serif", color="#f3f4f6"),
                        title_font=dict(family="Outfit, sans-serif", size=18, color="#ffffff"),
                        xaxis=dict(
                            gridcolor="rgba(255,255,255,0.05)",
                            title=dict(font=dict(size=14, family="Outfit, sans-serif"))
                        ),
                        yaxis=dict(
                            gridcolor="rgba(255,255,255,0.05)",
                            title=dict(font=dict(size=14, family="Outfit, sans-serif"))
                        ),
                        legend=dict(
                            bgcolor="rgba(17,24,39,0.8)",
                            bordercolor="rgba(255,255,255,0.08)",
                            borderwidth=1
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Metrics not yet available for visualization.")
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

        with tab_llm:
            traces = experiment.get("llm_traces", [])
            if traces:
                # High-level metrics
                total_tokens = sum(t["total_tokens"] for t in traces)
                total_cost = sum(t["cost"] for t in traces)
                avg_latency = sum(t["latency"] for t in traces) / len(traces) if traces else 0.0

                st.subheader("🤖 LLM Usage Summary")
                
                # Glassmorphic metrics row
                m_col1, m_col2, m_col3 = st.columns(3)
                with m_col1:
                    st.markdown(
                        f"""<div class="metric-card">
                            <div class="metric-value">{total_tokens:,}</div>
                            <div class="metric-label">Total Tokens Used</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                with m_col2:
                    st.markdown(
                        f"""<div class="metric-card">
                            <div class="metric-value">${total_cost:.5f}</div>
                            <div class="metric-label">Estimated Total Cost (USD)</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                with m_col3:
                    st.markdown(
                        f"""<div class="metric-card">
                            <div class="metric-value">{avg_latency:.2f}s</div>
                            <div class="metric-label">Average Response Latency</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )

                st.markdown("---")
                
                # Create visualizations
                st.subheader("📊 Observability Analytics")
                
                df_raw = pd.DataFrame(traces)
                
                # Cost Distribution
                df_cost = df_raw.groupby("agent_name")["cost"].sum().reset_index()
                df_cost.columns = ["Agent", "Cost"]
                
                # Avg Latency
                df_latency = df_raw.groupby("agent_name")["latency"].mean().reset_index()
                df_latency.columns = ["Agent", "Latency"]
                
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    fig_cost = px.pie(
                        df_cost,
                        names="Agent",
                        values="Cost",
                        hole=0.4,
                        title="LLM Budget Distribution (USD)",
                        color_discrete_sequence=px.colors.sequential.Purples_r
                    )
                    fig_cost.update_traces(textposition='inside', textinfo='percent+label')
                    fig_cost.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Inter, sans-serif", color="#f3f4f6"),
                        title_font=dict(family="Outfit, sans-serif", size=16, color="#ffffff"),
                        legend=dict(font=dict(size=10))
                    )
                    st.plotly_chart(fig_cost, use_container_width=True)
                    
                with chart_col2:
                    fig_latency = px.bar(
                        df_latency.sort_values("Latency"),
                        x="Latency",
                        y="Agent",
                        orientation="h",
                        title="Average Response Latency per Agent",
                        labels={"Latency": "Latency (s)", "Agent": "Agent"},
                        color="Latency",
                        color_continuous_scale="Purples"
                    )
                    fig_latency.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Inter, sans-serif", color="#f3f4f6"),
                        title_font=dict(family="Outfit, sans-serif", size=16, color="#ffffff"),
                        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                        coloraxis_showscale=False
                    )
                    st.plotly_chart(fig_latency, use_container_width=True)

                st.markdown("---")
                st.subheader("🔍 Execution Call Traces")

                rows = []
                for t in traces:
                    dt_str = t["created_at"]
                    if "T" in dt_str:
                        time_part = dt_str.split("T")[1].split(".")[0]
                    else:
                        time_part = dt_str

                    rows.append({
                        "Agent": t["agent_name"],
                        "Prompt Tokens": t["prompt_tokens"],
                        "Completion Tokens": t["completion_tokens"],
                        "Total Tokens": t["total_tokens"],
                        "Cost": f"${t['cost']:.6f}",
                        "Latency": f"{t['latency']:.3f}s",
                        "Time": time_part
                    })

                df_traces = pd.DataFrame(rows)
                st.dataframe(df_traces, use_container_width=True, hide_index=True)
            else:
                st.info("No LLM traces recorded yet for this experiment.")
    else:
        st.warning("Could not load experiment details.")
else:
    # Landing page
    st.markdown("### 🎯 How It Works")
    st.markdown(
        """
        <div class="agent-grid">
            <div class="agent-card">
                <div class="agent-title">1. Describe Goal</div>
                <div class="agent-role">Specify what you want to predict in plain English (e.g., tumor diagnosis, diabetes progression).</div>
            </div>
            <div class="agent-card">
                <div class="agent-title">2. Select Data</div>
                <div class="agent-role">Select one of our high-quality built-in datasets or provide your own absolute path to a CSV.</div>
            </div>
            <div class="agent-card">
                <div class="agent-title">3. Launch & Watch</div>
                <div class="agent-role">The autonomous system triggers 9 agents to collaborate, plan, evaluate, tune, and report results.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🤖 Multi-Agent Architecture")
    st.markdown(
        """
        <div class="agent-grid">
            <div class="agent-card">
                <div class="agent-title">🎯 Planner</div>
                <div class="agent-role">Translates goals into experiment plans, selecting models, primary metrics, and standard evaluation targets.</div>
            </div>
            <div class="agent-card">
                <div class="agent-title">📊 Data Analyst</div>
                <div class="agent-role">Profiles the input dataset, diagnosing missing ratios, schema details, class imbalances, and feature distributions.</div>
            </div>
            <div class="agent-card">
                <div class="agent-title">🔧 Feature Engineer</div>
                <div class="agent-role">Designs pipeline preprocessing configurations and dynamically compiles custom feature synthesis scripts.</div>
            </div>
            <div class="agent-card">
                <div class="agent-title">🏃 Runner</div>
                <div class="agent-role">Runs data split pipelines, trains modeling algorithms, and logs all experiments to MLflow.</div>
            </div>
            <div class="agent-card">
                <div class="agent-title">⚡ HPO Agent</div>
                <div class="agent-role">Executes Bayesian sweeps using Optuna, optimizing model configurations across trial parameters.</div>
            </div>
            <div class="agent-card">
                <div class="agent-title">📈 Evaluator</div>
                <div class="agent-role">Gathers modeling runs, computes metrics, ranks performance, and selects the current champion candidate.</div>
            </div>
            <div class="agent-card">
                <div class="agent-title">🔍 Critic</div>
                <div class="agent-role">Audits modeling performance, checks convergence, looks for overfitting, and decides if extra loops are needed.</div>
            </div>
            <div class="agent-card">
                <div class="agent-title">🔄 Replanner</div>
                <div class="agent-role">Reformulates strategies on failures, adjusting parameters or revising engineering plans based on critic audits.</div>
            </div>
            <div class="agent-card">
                <div class="agent-title">📝 Reporter</div>
                <div class="agent-role">Compiles technical Markdown reports summarizing workflow analysis, feature additions, and tuning graphs.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #9ca3af;'>👈 Launch your first experiment using the sidebar to see the orchestrator in action!</p>", unsafe_allow_html=True)
