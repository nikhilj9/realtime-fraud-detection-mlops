# Dockerfile.mlflow
# ============================================================================
# Custom MLflow Server Image
# ============================================================================
# WHY BUILD OUR OWN?
#   - All dependencies pre-installed (no runtime pip install)
#   - Starts in seconds, not minutes
#   - No network dependency during pod startup
#   - 100% reproducible
# ============================================================================

# Dockerfile.mlflow
FROM python:3.12-slim

# Install MLflow 3.6.0 and the production database/S3 drivers
RUN pip install --no-cache-dir \
    mlflow==3.6.0 \
    psycopg2-binary \
    boto3

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]