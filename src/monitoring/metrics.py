"""Prometheus metric definitions."""

from prometheus_client import Counter, Histogram

# Business Metrics
PREDICTION_TOTAL = Counter(
    "fraud_prediction_total",
    "Total number of predictions made",
    labelnames=["status", "risk_level"]
)

# Performance Metrics
PREDICTION_LATENCY = Histogram(
    "fraud_prediction_latency_seconds",
    "Time spent processing prediction requests",
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]  # Focus on sub-200ms
)