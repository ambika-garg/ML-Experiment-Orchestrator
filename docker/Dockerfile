FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

# Copy app code
COPY . .

# Expose API port
EXPOSE 8000

# Run with uvicorn
CMD ["uvicorn", "ml_experiment_orchestrator.main:app", \
     "--host", "0.0.0.0", "--port", "8000"]
