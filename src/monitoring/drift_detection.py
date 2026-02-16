"""
Core drift detection logic using Evidently AI 0.7.18.

Key API patterns for 0.7.18:
- Report is a configuration class
- report.run() returns a Snapshot object
- Use snapshot.json() to get results (NOT snapshot.as_dict())
- JSON structure: metrics[].metric_name, metrics[].value
- Column names are embedded in metric_name string: "ValueDrift(column=X,...)"
"""

import gc
import json
import os
import re
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from src.monitoring.drift_config import (
    ALL_MONITORED_COLUMNS,
    CATEGORICAL_COLUMNS,
    DRIFT_THRESHOLDS,
    NUMERICAL_COLUMNS,
    DriftSeverity,
    DriftThresholds,
)
from src.utils.logger import get_logger

# Suppress SciPy "divide by zero" warning in Chi-Squared test.
# This is expected behavior when new categories appear (drift).
# It produces 'inf' stats, which correctly results in p-value = 0 (drift detected).
warnings.filterwarnings(
    "ignore",
    message="divide by zero encountered in divide",
    category=RuntimeWarning,
)

logger = get_logger(__name__)


@dataclass
class DriftDetectionResult:
    """
    Result of drift detection analysis.

    Attributes:
        severity: Overall drift severity (NONE, WARNING, CRITICAL)
        drifted_columns: List of column names that showed drift
        total_columns: Total number of columns analyzed
        drift_ratio: Proportion of columns with drift (0.0 to 1.0)
        column_scores: Dict mapping column name to p-value
        report_path: Path to saved HTML report (if saved)
    """
    severity: DriftSeverity
    drifted_columns: list[str]
    total_columns: int
    drift_ratio: float
    column_scores: dict[str, float]
    report_path: Path | None = None


def create_data_definition() -> DataDefinition:
    """
    Create Evidently DataDefinition from our column configuration.

    Maps numerical and categorical columns for appropriate
    statistical tests (KS for numerical, chi-squared for categorical).
    """
    return DataDefinition(
        numerical_columns=NUMERICAL_COLUMNS,
        categorical_columns=CATEGORICAL_COLUMNS,
    )


def create_datasets(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> tuple[Dataset, Dataset]:
    """
    Create Evidently Dataset objects from pandas DataFrames.

    Filters to only monitored columns to avoid errors with
    ID columns, timestamps, etc.
    """
    data_definition = create_data_definition()

    existing_cols = set(reference_df.columns)
    monitored_cols = [c for c in ALL_MONITORED_COLUMNS if c in existing_cols]

    logger.debug(f"Creating datasets with {len(monitored_cols)} monitored columns")

    reference_dataset = Dataset.from_pandas(
        reference_df[monitored_cols],
        data_definition=data_definition,
    )
    current_dataset = Dataset.from_pandas(
        current_df[monitored_cols],
        data_definition=data_definition,
    )

    return current_dataset, reference_dataset


def parse_drift_from_json(json_str: str, threshold: float = 0.05) -> dict[str, Any]:
    """
    Parse Evidently 0.7.18 JSON structure to extract drift information.

    In 0.7.18, the JSON structure is:
    {
        "metrics": [
            {
                "metric_name": "ValueDrift(column=X,method=...,threshold=0.05)",
                "value": 0.001  # p-value as float
            },
            ...
        ]
    }

    Args:
        json_str: Raw JSON string from snapshot.json()
        threshold: P-value threshold for drift detection (default: 0.05)

    Returns:
        dict with keys:
            - 'total_columns': int
            - 'drifted_columns': list[str]
            - 'column_scores': dict[str, float]
    """
    data = json.loads(json_str)
    metrics = data.get("metrics", [])

    result = {
        "total_columns": 0,
        "drifted_columns": [],
        "column_scores": {},
    }

    for m in metrics:
        metric_name = m.get("metric_name", "")
        value = m.get("value")

        # Extract per-column drift from ValueDrift metrics
        if "ValueDrift" in metric_name:
            # Parse column name from: "ValueDrift(column=amount,method=...)"
            match = re.search(r"column=([^,\)]+)", metric_name)
            if match:
                col_name = match.group(1)

                # Extract threshold from metric_name if present
                threshold_match = re.search(r"threshold=([0-9.]+)", metric_name)
                col_threshold = float(threshold_match.group(1)) if threshold_match else threshold

                # Value is the p-value (float)
                p_value = float(value) if value is not None else 1.0

                result["column_scores"][col_name] = p_value

                # Drift detected if p-value < threshold
                if p_value < col_threshold:
                    result["drifted_columns"].append(col_name)

    result["total_columns"] = len(result["column_scores"])

    return result


def detect_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    save_report: bool = True,
    output_dir: str | Path = "reports",
    thresholds: DriftThresholds = DRIFT_THRESHOLDS,
) -> DriftDetectionResult:
    """
    Main entry point for drift detection.

    Compares current data distribution against reference (baseline) data
    and determines drift severity based on the proportion of drifted columns.

    Args:
        reference_df: Baseline data (e.g., training data distribution)
        current_df: Current data to compare against baseline
        save_report: Whether to save HTML report
        output_dir: Directory for HTML reports
        thresholds: Threshold configuration for severity levels

    Returns:
        DriftDetectionResult with severity, drifted columns, and scores

    Example:
        >>> result = detect_drift(training_df, production_df)
        >>> if result.severity == DriftSeverity.CRITICAL:
        ...     raise ValueError("Critical drift detected!")
    """
    logger.info("=" * 60)
    logger.info("DRIFT DETECTION (Evidently 0.7.18)")
    logger.info("=" * 60)
    logger.info(f"Reference data: {len(reference_df)} rows")
    logger.info(f"Current data: {len(current_df)} rows")

    # 1. Create Evidently Datasets
    current_ds, reference_ds = create_datasets(reference_df, current_df)

    # 2. Configure and Run Report
    report = Report([DataDriftPreset()])

    logger.info("Running drift analysis...")
    snapshot = report.run(reference_data=reference_ds, current_data=current_ds)
    logger.info("Drift analysis complete")

    # Free Evidently datasets immediately - they're no longer needed
    del current_ds, reference_ds, report
    gc.collect()

    # 3. Parse Results from Snapshot JSON
    drift_info = parse_drift_from_json(snapshot.json())

    total_columns = drift_info["total_columns"]
    drifted_columns = drift_info["drifted_columns"]
    column_scores = drift_info["column_scores"]

    # 4. Calculate Severity
    drift_ratio = len(drifted_columns) / total_columns if total_columns > 0 else 0.0

    if drift_ratio >= thresholds.critical:
        severity = DriftSeverity.CRITICAL
    elif drift_ratio >= thresholds.warning:
        severity = DriftSeverity.WARNING
    else:
        severity = DriftSeverity.NONE

    # 5. Save Report (JSON always, HTML only if not in K8s)
    report_path = None
    if save_report:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

        # Always save JSON (lightweight, ~100KB)
        json_report_path = output_path / f"drift_report_{timestamp}.json"
        with open(json_report_path, "w") as f:
            f.write(snapshot.json())
        logger.info(f"JSON report saved: {json_report_path}")
        report_path = json_report_path

        # HTML is memory-intensive (~2-3GB spike for 41 columns)
        # Only generate if DRIFT_SAVE_HTML=true (skip in K8s by default)
        if os.getenv("DRIFT_SAVE_HTML", "false").lower() == "true":
            html_report_path = output_path / f"drift_report_{timestamp}.html"
            snapshot.save_html(str(html_report_path))
            report_path = html_report_path
            logger.info(f"HTML report saved: {html_report_path}")
        else:
            logger.info("HTML report skipped (set DRIFT_SAVE_HTML=true to enable)")

    # 6. Log Summary
    logger.info("Drift Summary:")
    logger.info(f"  - Columns analyzed: {total_columns}")
    logger.info(f"  - Columns drifted: {len(drifted_columns)}")
    logger.info(f"  - Drift ratio: {drift_ratio:.1%}")
    logger.info(f"  - Severity: {severity.value.upper()}")

    if drifted_columns:
        display_cols = drifted_columns[:5]
        suffix = f"... (+{len(drifted_columns) - 5} more)" if len(drifted_columns) > 5 else ""
        logger.info(f"  - Drifted: {display_cols}{suffix}")

    logger.info("=" * 60)

    return DriftDetectionResult(
        severity=severity,
        drifted_columns=drifted_columns,
        total_columns=total_columns,
        drift_ratio=drift_ratio,
        column_scores=column_scores,
        report_path=report_path,
    )
