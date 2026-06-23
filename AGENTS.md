# ML Experiment Orchestrator - Agents Guide

## Project Overview

The ML Experiment Orchestrator is an autonomous multi-agent system that acts as an AI Research Scientist. It automatically plans, executes, evaluates, and improves machine learning experiments based on natural language goals.

The system uses 9 specialised agents orchestrated via LangGraph, simulating a full data science lifecycle.

## Stack

- **Framework:** LangGraph + Gemini 2.5 Flash
- **ML & Data Science:** scikit-learn, XGBoost, Optuna, Pandas
- **Experiment Tracking:** MLflow
- **API Backend:** FastAPI + Uvicorn
- **Database:** PostgreSQL (Production) / SQLite (Development) using SQLAlchemy ORM
- **UI:** Streamlit
- **Infra:** Docker Compose

## Architecture & Conventions

### Multi-Agent System

The orchestration graph consists of the following 9 agents:

1. **Planner (`PlannerAgent`)**: LLM-based agent that understands goals and selects algorithms & metrics.
2. **Data Analyst (`DataAnalysisAgent`)**: LLM + Code agent that profiles datasets and detects issues (missing values, imbalance).
3. **Feature Engineer (`FeatureEngineeringAgent`)**: LLM-based agent that designs preprocessing pipelines.
4. **Runner (`ExperimentRunnerAgent`)**: Code-based agent that executes preprocessing, trains models, and logs to MLflow.
5. **HPO (`HyperparameterOptimizationAgent`)**: Code-based agent that performs Optuna hyperparameter tuning.
6. **Evaluator (`EvaluationAgent`)**: LLM + Code agent that ranks models and computes evaluation metrics.
7. **Critic (`CriticAgent`)**: LLM-based agent that identifies weaknesses, proposes improvements, and checks for convergence.
8. **Replanner (`ReplannerAgent`)**: LLM-based agent that generates improved experiment plans based on Critic's feedback.
9. **Reporter (`ReportAgent`)**: LLM-based agent that creates comprehensive Markdown reports.

### Code Conventions

- **State Management**: LangGraph `ExperimentState` (a TypedDict) is passed between all agent nodes. It uses an append-reducer for `experiment_results` to accumulate metrics across iterations.
- **Lazy Initialization**: LLM instances in agents are lazily initialized on first use, allowing modules to be imported without requiring an API key immediately.
- **Asynchronous Processing**: FastAPI routes use background tasks (`BackgroundTasks`) to launch experiments so the API remains non-blocking. Database sessions are fully async using SQLAlchemy's async engine.

## Run Commands

### Local Development Setup

```bash
# 1. Set up a virtual environment (Python 3.11+)
python3.11 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -e ".[dev]"

# 3. Set environment variables
cp .env.example .env
# Edit .env and set GOOGLE_API_KEY=your-gemini-api-key

# 4. Start the API server
python -m ml_experiment_orchestrator.main

# 5. Start the Streamlit UI (in a new terminal)
source venv/bin/activate
streamlit run ui/app.py
```

### Docker Setup

```bash
export GOOGLE_API_KEY=your-key
docker compose -f docker/docker-compose.yml up --build
```
