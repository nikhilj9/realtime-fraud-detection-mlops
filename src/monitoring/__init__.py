"""
Drift monitoring module for fraud detection system.

Provides data drift detection using Evidently AI 0.7.18+.

Usage:
    from src.monitoring import detect_drift, DriftSeverity, DriftDetectionResult
    
    result = detect_drift(reference_df, current_df)
    if result.severity == DriftSeverity.CRITICAL:
        raise ValueError("Critical drift!")
"""

from src.monitoring.drift_config import (
    DriftSeverity as DriftSeverity,
    DriftThresholds as DriftThresholds,
    DRIFT_THRESHOLDS as DRIFT_THRESHOLDS,
    NUMERICAL_COLUMNS as NUMERICAL_COLUMNS,
    CATEGORICAL_COLUMNS as CATEGORICAL_COLUMNS,
    KEY_FEATURES as KEY_FEATURES,
)

from src.monitoring.drift_detection import (
    detect_drift as detect_drift,
    DriftDetectionResult as DriftDetectionResult,
)

from src.monitoring.drift_simulation import (
    apply_drift_scenario as apply_drift_scenario,
    split_data_for_drift as split_data_for_drift,
    DriftType as DriftType,
)