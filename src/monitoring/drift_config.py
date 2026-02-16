"""
Drift detection configuration.

Centralizes thresholds, column definitions, and severity levels
for drift monitoring across the fraud detection system.
"""

from dataclasses import dataclass
from enum import Enum


class DriftSeverity(Enum):
    """Drift severity levels for pipeline decision-making."""

    NONE = "none"          # No significant drift detected
    WARNING = "warning"    # Some drift, but acceptable - log and continue
    CRITICAL = "critical"  # Severe drift - should block pipeline


@dataclass(frozen=True)
class DriftThresholds:
    """
    Thresholds for drift severity classification.

    Based on proportion of columns showing statistical drift.
    Uses frozen=True for immutability (production safety).
    """

    warning: float = 0.2   # >20% columns drifted = WARNING
    critical: float = 0.5  # >50% columns drifted = CRITICAL


# Default thresholds instance
DRIFT_THRESHOLDS = DriftThresholds()


# =============================================================================
# Column Definitions for Drift Analysis
# =============================================================================
# Based on transactions_enriched.parquet schema (46 columns)
# Matches PROCESSED_DATA_COLUMNS from src/validation/expectations.py

# ID columns - excluded from drift analysis (unique identifiers)
ID_COLUMNS: list[str] = [
    "transaction_id",
    "user_id",
    "merchant_id",
]

# Timestamp columns - excluded (time-based, expected to change)
DATETIME_COLUMNS: list[str] = [
    "timestamp",
]

# Target column - tracked separately for prediction drift
TARGET_COLUMN: str = "is_fraud"

# Numerical columns - continuous values, use KS test for drift
NUMERICAL_COLUMNS: list[str] = [
    # Transaction attributes
    "credit_limit",
    "card_age",
    "amount_inr",
    # PCA components (V1-V28 from original creditcard.csv)
    "V1", "V2", "V3", "V4", "V5", "V6", "V7",
    "V8", "V9", "V10", "V11", "V12", "V13", "V14",
    "V15", "V16", "V17", "V18", "V19", "V20", "V21",
    "V22", "V23", "V24", "V25", "V26", "V27", "V28",
]

# Categorical columns - discrete values, use chi-squared test for drift
CATEGORICAL_COLUMNS: list[str] = [
    "card_network",
    "card_issuer",
    "card_tier",
    "merchant_category",
    "merchant_city",
    "merchant_state",
    "transaction_channel",
    "entry_mode",
    "is_international",
    "is_recurring",
]

# Key features - business-critical columns to highlight in reports
# Drift in these columns is especially important for fraud detection
KEY_FEATURES: list[str] = [
    "amount_inr",           # Transaction size
    "merchant_category",    # Spending patterns
    "transaction_channel",  # Online vs POS behavior
    "is_international",     # Geography risk factor
]

# All columns to monitor (excluding IDs, timestamp, target)
ALL_MONITORED_COLUMNS: list[str] = NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS


def get_column_counts() -> dict[str, int]:
    """Return counts of each column type for logging/debugging."""
    return {
        "numerical": len(NUMERICAL_COLUMNS),
        "categorical": len(CATEGORICAL_COLUMNS),
        "total_monitored": len(ALL_MONITORED_COLUMNS),
        "excluded_ids": len(ID_COLUMNS),
        "excluded_datetime": len(DATETIME_COLUMNS),
    }
