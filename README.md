# 🧪 ML Experiment Orchestrator

> An autonomous multi-agent system that acts as an AI Research Scientist — planning, executing, evaluating, and improving ML experiments.

## Architecture

```
User Goal → 🎯 Planner → 📊 Data Analysis → 🔧 Feature Engineering
    → 🏃 Experiment Runner → 📈 Evaluation → 🔍 Critic
    → 🔄 Replanner → 🏃 Runner (loop) → 📝 Report
```

The system uses **9 specialised agents** orchestrated via [LangGraph](https://github.com/langchain-ai/langgraph):

| Agent | Type | Role |
|-------|------|------|
| 🎯 Planner | LLM | Understands goals, selects algorithms & metrics |
| 📊 Data Analyst | LLM + Code | Profiles datasets, detects issues |
| 🔧 Feature Engineer | LLM | Designs preprocessing pipelines |
| 🏃 Runner | Code | Trains models, logs to MLflow |
| ⚡ HPO | Code | Optuna hyperparameter tuning |
| 📈 Evaluator | LLM + Code | Ranks models, interprets results |
| 🔍 Critic | LLM | Identifies weaknesses, proposes improvements |
| 🔄 Replanner | LLM | Generates improved experiment plans |
| 📝 Reporter | LLM | Creates comprehensive Markdown reports |

## Tech Stack

- **Framework:** LangGraph + Gemini 2.5 Flash
- **ML:** scikit-learn, XGBoost, Optuna
- **Tracking:** MLflow
- **API:** FastAPI + Uvicorn
- **Database:** PostgreSQL (prod) / SQLite (dev)
- **UI:** Streamlit
- **Infra:** Docker Compose

## Quick Start

### 1. Prerequisites

- Python 3.11+
- A Google API key for Gemini, unless running in demo mode

### 2. Setup

```bash
# Clone and enter
cd ML-Experiment-Orchestraor

# Copy env template
cp .env.example .env
# For live LLM agents, set GOOGLE_API_KEY=your-key
# For local demos without an API key, set DEMO_MODE=true

# Install dependencies
pip install -e ".[dev]"
```

### 3. Run (Local Dev)

```bash
# Start the API server
python -m ml_experiment_orchestrator.main

# In another terminal — start the Streamlit UI
streamlit run ui/app.py
```

### 4. Run Without an API Key

Demo mode uses deterministic local responses for the LLM-backed agents. The
workflow still loads data, trains models, logs runs, evaluates metrics, and
generates a report.

```bash
DEMO_MODE=true python -m ml_experiment_orchestrator.main
```

Then launch the UI in another terminal:

```bash
DEMO_MODE=true streamlit run ui/app.py
```

### 5. Run (Docker)

```bash
# Set your API key
export GOOGLE_API_KEY=your-key

# Start the full stack
docker compose -f docker/docker-compose.yml up --build
```

Services:
- **API:** http://localhost:8000 (docs at /docs)
- **MLflow:** http://localhost:5001
- **Streamlit:** http://localhost:8501

## API Usage

### Create an Experiment

```bash
curl -X POST http://localhost:8000/api/v1/experiment \
  -H "Content-Type: application/json" \
  -d '{"goal": "Predict heart disease from patient clinical data", "dataset": "heart_disease"}'
```

### Check Status

```bash
curl http://localhost:8000/api/v1/experiment/{id}
```

### Get Report

```bash
curl http://localhost:8000/api/v1/experiment/{id}/report
```

### List Runs

```bash
curl http://localhost:8000/api/v1/experiment/{id}/runs
```

## Built-in Datasets

| Name | Description | Task |
|------|-------------|------|
| `heart_disease` | UCI Heart Disease (Statlog) | Classification |
| `breast_cancer` | Wisconsin Breast Cancer | Classification |
| `diabetes` | Diabetes progression | Regression |

## Project Structure

```
ml_experiment_orchestrator/
├── agents/           # All 9 agent implementations
├── graph/            # LangGraph state, routing, workflow
├── models/           # SQLAlchemy ORM + Pydantic schemas
├── services/         # MLflow, model registry, data loader
├── api/              # FastAPI routes + dependencies
├── db/               # Database session management
├── main.py           # App entry point
└── config.py         # Pydantic settings
ui/                   # Streamlit dashboard
docker/               # Dockerfiles + docker-compose
tests/                # Test suite
```

## How It Works

1. **User submits a goal** → "Predict heart disease from patient data"
2. **Planner** determines: classification problem, try LogisticRegression/RF/XGBoost, measure F1/ROC-AUC
3. **Data Analyst** profiles the dataset: 303 rows, 13 features, 1.5% missing, slight class imbalance
4. **Feature Engineer** designs: median imputation, standard scaling, optional SMOTE
5. **Runner** trains all 3 models with train/test split, logs to MLflow
6. **Evaluator** ranks models: XGBoost (0.89) > RF (0.87) > LR (0.84)
7. **Critic** suggests: try SMOTE for class imbalance, tune XGBoost hyperparameters
8. **Replanner** creates updated plan with SMOTE + HPO
9. **Runner** re-trains with improvements → XGBoost HPO (0.92)
10. **Critic** sees convergence → stops
11. **Reporter** generates comprehensive Markdown report

## License

MIT
