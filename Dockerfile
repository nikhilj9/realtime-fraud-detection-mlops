# ============================================
# Builder Stage
# ============================================
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Enable bytecode compilation only if you prioritize start speed over image size
# Since this is a worker, we'll keep size low.
ENV UV_COMPILE_BYTECODE=0
ENV UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./

# Use a cache mount to speed up subsequent builds without bloating the image
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# ============================================
# Runtime Stage
# ============================================
FROM python:3.12-slim

WORKDIR /app

# 1. Install system dependencies (curl for health checks, libgomp1 for sklearn/XGBoost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy the virtual environment from builder
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# 3. Application code only (artifacts loaded from PVC/S3 at runtime)
COPY src/ src/

# 4. Prefect specific configuration
# Ensure logs are sent to the console immediately (don't buffer)
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Since your previous container was a "prefect worker start"
# You should decide if this image is for the API or the WORKER.
# If it's for the API, keep your uvicorn command:
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

# Make sure prefect is installed in your uv sync groups
# If this image is ONLY for the worker, change the CMD:
# CMD ["prefect", "worker", "start", "--pool", "default-agent-pool", "--type", "kubernetes"]