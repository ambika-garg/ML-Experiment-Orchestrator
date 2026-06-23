"""Application configuration via environment variables.

Uses pydantic-settings to load configuration from .env files and environment
variables with type validation and sensible defaults.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────────────
    google_api_key: str = ""
    llm_model: str = "gemini-2.5-flash"
    llm_temperature: float = 0.2
    demo_mode: bool = False

    # ── Database ─────────────────────────────────────────────────────────
    database_url: str = "sqlite+aiosqlite:///./ml_orchestrator.db"

    # ── MLflow ───────────────────────────────────────────────────────────
    mlflow_tracking_uri: str = "mlruns"

    # ── Experiment Defaults ──────────────────────────────────────────────
    max_iterations: int = 3
    convergence_threshold: float = 0.01
    hpo_n_trials: int = 50
    cv_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42

    # ── API ──────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ── Data ─────────────────────────────────────────────────────────────
    data_dir: str = "./data"


# Singleton instance — import this everywhere
settings = Settings()
