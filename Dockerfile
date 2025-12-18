# Builder

FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./

# Install dependencies
# --frozen: strict adherence to uv.lock
# --no-dev: exclude testing/dev dependencies
# --no-install-project: we only want dependencies, not our code yet
RUN uv sync --frozen --no-dev --no-install-project

# Runtime 

FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH"

# Copy Application Code
COPY src/ src/

# 1. The Model
COPY models/champion_model.joblib models/champion_model.joblib

# 2. The Encoders/Metadata
COPY data/processed/target_encoder.joblib data/processed/target_encoder.joblib
COPY data/processed/feature_columns.joblib data/processed/feature_columns.joblib

# Create logs directory
RUN mkdir logs

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]