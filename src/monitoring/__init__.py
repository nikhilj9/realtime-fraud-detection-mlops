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
    CATEGORICAL_COLUMNS as CATEGORICAL_COLUMNS,
)
from src.monitoring.drift_config import (
    DRIFT_THRESHOLDS as DRIFT_THRESHOLDS,
)
from src.monitoring.drift_config import (
    KEY_FEATURES as KEY_FEATURES,
)
from src.monitoring.drift_config import (
    NUMERICAL_COLUMNS as NUMERICAL_COLUMNS,
)
from src.monitoring.drift_config import (
    DriftSeverity as DriftSeverity,
)
from src.monitoring.drift_config import (
    DriftThresholds as DriftThresholds,
)
from src.monitoring.drift_detection import (
    DriftDetectionResult as DriftDetectionResult,
)
from src.monitoring.drift_detection import (
    detect_drift as detect_drift,
)
from src.monitoring.drift_simulation import (
    DriftType as DriftType,
)
from src.monitoring.drift_simulation import (
    apply_drift_scenario as apply_drift_scenario,
)
from src.monitoring.drift_simulation import (
    split_data_for_drift as split_data_for_drift,
)
