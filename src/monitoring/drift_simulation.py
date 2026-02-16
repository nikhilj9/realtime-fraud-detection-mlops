"""
Artificial drift simulation for testing monitoring systems.

Provides utilities to inject various types of drift (numerical, categorical,
nulls) into datasets to validate drift detection logic.
"""

from typing import Any, Literal

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


DriftType = Literal["none", "mild", "severe"]


def split_data_for_drift(
    df: pd.DataFrame,
    reference_ratio: float = 0.5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into reference (baseline) and current (production) sets.

    Args:
        df: Source dataframe
        reference_ratio: Proportion of data to keep as reference (0.0-1.0)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (reference_df, current_df)
    """
    # Sort by time if available to simulate temporal split
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")
        split_idx = int(len(df) * reference_ratio)
        reference_df = df.iloc[:split_idx]
        current_df = df.iloc[split_idx:].copy()
        logger.info(f"Split data by time: {len(reference_df)} ref, {len(current_df)} curr")
    else:
        # Random split if no time
        logger.warning("No timestamp column found, using random split")
        shuffled = df.sample(frac=1, random_state=seed)
        split_idx = int(len(df) * reference_ratio)
        reference_df = shuffled.iloc[:split_idx].copy()
        current_df = shuffled.iloc[split_idx:].copy()

    return reference_df, current_df


def inject_numerical_drift(
    df: pd.DataFrame,
    column: str,
    multiplier: float = 1.0,
    offset: float = 0.0,
) -> pd.DataFrame:
    """
    Shift numerical distribution by multiplying and/or adding offset.

    new_value = (old_value * multiplier) + offset
    """
    if column not in df.columns:
        logger.warning(f"Column {column} not found, skipping drift injection")
        return df

    df = df.copy()
    df[column] = (df[column] * multiplier) + offset
    logger.debug(f"Injected numerical drift on {column}: x{multiplier} + {offset}")
    return df


def inject_categorical_drift(
    df: pd.DataFrame,
    column: str,
    target_category: Any,
    probability: float,
) -> pd.DataFrame:
    """
    Replace values in a column with target_category with given probability.
    Simulates a category becoming more dominant.

    Handles dtype conversion properly to avoid FutureWarning.
    """
    if column not in df.columns:
        logger.warning(f"Column {column} not found, skipping drift injection")
        return df

    df = df.copy()
    mask = np.random.random(len(df)) < probability

    # Convert column to object dtype first to avoid incompatible dtype warning
    # This is the proper way to handle mixed types in pandas 2.x
    if df[column].dtype != object:
        df[column] = df[column].astype(object)

    df.loc[mask, column] = target_category

    logger.debug(f"Injected categorical drift on {column}: {probability:.1%} -> {target_category}")
    return df


def inject_nulls(
    df: pd.DataFrame,
    column: str,
    missing_ratio: float,
) -> pd.DataFrame:
    """Introduce missing values (NaN/None) into a column."""
    if column not in df.columns:
        logger.warning(f"Column {column} not found, skipping drift injection")
        return df

    df = df.copy()
    mask = np.random.random(len(df)) < missing_ratio
    df.loc[mask, column] = None
    logger.debug(f"Injected nulls on {column}: {missing_ratio:.1%}")
    return df


def apply_drift_scenario(
    df: pd.DataFrame,
    scenario: DriftType,
) -> pd.DataFrame:
    """
    Apply a predefined drift scenario to the dataset.

    Args:
        df: Input dataframe
        scenario: "none", "mild", or "severe"

    Returns:
        Drifted dataframe
    """
    if scenario == "none":
        return df

    df_drifted = df.copy()
    logger.info(f"Applying '{scenario}' drift scenario...")

    if scenario == "mild":
        # Slight shift in transaction amounts (5% increase)
        df_drifted = inject_numerical_drift(df_drifted, "amount_inr", multiplier=1.05)

        # Minor shift in channel usage
        df_drifted = inject_categorical_drift(
            df_drifted, "transaction_channel", "Online", 0.1
        )

    elif scenario == "severe":
        # Major economic shift (50% increase + offset)
        df_drifted = inject_numerical_drift(df_drifted, "amount_inr", multiplier=1.5, offset=500)

        # Data quality failure (20% missing cities)
        df_drifted = inject_nulls(df_drifted, "merchant_city", 0.2)

        # Massive behavior change (40% shift to International)
        df_drifted = inject_categorical_drift(
            df_drifted, "is_international", True, 0.4
        )

        # Feature drift in PCA components (simulating core pattern change)
        df_drifted = inject_numerical_drift(df_drifted, "V1", multiplier=1.2, offset=0.5)
        df_drifted = inject_numerical_drift(df_drifted, "V4", multiplier=0.8, offset=-0.5)

    return df_drifted
