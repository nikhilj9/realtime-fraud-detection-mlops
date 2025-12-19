"""Feature engineering with proper train/val/test split."""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib

from src.features.encoders import TargetEncoder
from src.utils import get_logger

logger = get_logger(__name__)

# =============================================================================
# TIME-BASED SPLIT (3-WAY)
# =============================================================================

def time_based_split(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    timestamp_col: str = "timestamp"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train/validation/test by time."""
    
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    # Validate no time leakage
    assert train_df[timestamp_col].max() <= val_df[timestamp_col].min(), "Leakage: train/val"
    assert val_df[timestamp_col].max() <= test_df[timestamp_col].min(), "Leakage: val/test"
    
    logger.info(f"Split: Train={len(train_df):,} | Val={len(val_df):,} | Test={len(test_df):,}")
    logger.info(f"Train: {train_df[timestamp_col].min()} to {train_df[timestamp_col].max()}")
    logger.info(f"Val:   {val_df[timestamp_col].min()} to {val_df[timestamp_col].max()}")
    logger.info(f"Test:  {test_df[timestamp_col].min()} to {test_df[timestamp_col].max()}")
    
    return train_df, val_df, test_df


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def add_time_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> None:
    """Add time-based features."""
    ts = pd.to_datetime(df[timestamp_col])
    
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["is_night"] = ts.dt.hour.isin([0, 1, 2, 3, 4, 5]).astype(int)
    df["is_weekend"] = ts.dt.dayofweek.isin([5, 6]).astype(int)


def add_amount_features(df: pd.DataFrame, amount_col: str = "amount_inr") -> None:
    """Add amount-based features."""
    df["log_amount"] = np.log1p(df[amount_col])


def add_interaction_features(df: pd.DataFrame) -> None:
    """Add interaction features."""
    df["intl_online"] = ((df["is_international"] == 1) & (df["transaction_channel"] == "Online")).astype(int)
    
    if "is_night" in df.columns and "log_amount" in df.columns:
        df["night_high_amount"] = ((df["is_night"] == 1) & (df["log_amount"] > df["log_amount"].quantile(0.9))).astype(int)


def engineer_features(
    df: pd.DataFrame,
    target_encoder: TargetEncoder = None,
    fit_encoder: bool = False,
    y: pd.Series = None
) -> Tuple[pd.DataFrame, TargetEncoder]:
    """Apply all feature engineering."""
    
    df = df.copy()
    
    add_time_features(df)
    add_amount_features(df)
    add_interaction_features(df)
    
    if fit_encoder and y is not None:
        target_encoder = TargetEncoder(column="merchant_category")
        target_encoder.fit(df, y)
    
    if target_encoder:
        df = target_encoder.transform(df)
    
    return df, target_encoder


# =============================================================================
# FEATURE SELECTION
# =============================================================================

def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get final feature columns for modeling."""
    
    exclude = [
        "transaction_id", "user_id", "merchant_id", "timestamp",
        "card_network", "card_issuer", "entry_mode",
        "merchant_category", "merchant_city", "merchant_state",
        "is_fraud"
    ]
    
    return [c for c in df.columns if c not in exclude]


def prepare_modeling_data(
    df: pd.DataFrame,
    feature_cols: List[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract X and y for modeling."""
    
    X = df[feature_cols].copy()
    y = df["is_fraud"].copy()
    
    return X, y


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_feature_engineering(
    input_path: Path,
    output_dir: Path,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2
) -> Dict[str, Path]:
    """Run complete feature engineering pipeline with 3-way split."""
    
    logger.info("=" * 50)
    logger.info("Starting Feature Engineering")
    logger.info("=" * 50)
    
    # Load data
    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df):,} rows")
    
    # Time-based split
    train_df, val_df, test_df = time_based_split(df, train_ratio, val_ratio)
    
    # Engineer features (fit ONLY on train)
    y_train = train_df["is_fraud"]
    train_df, target_encoder = engineer_features(train_df, fit_encoder=True, y=y_train)
    
    # Apply to val and test (transform only)
    val_df, _ = engineer_features(val_df, target_encoder=target_encoder)
    test_df, _ = engineer_features(test_df, target_encoder=target_encoder)
    
    # Get feature columns
    feature_cols = get_feature_columns(train_df)
    logger.info(f"Features: {len(feature_cols)} columns")
    
    # Prepare final datasets
    X_train, y_train = prepare_modeling_data(train_df, feature_cols)
    X_val, y_val = prepare_modeling_data(val_df, feature_cols)
    X_test, y_test = prepare_modeling_data(test_df, feature_cols)
    
    # Add target back
    train_out = X_train.copy()
    train_out["is_fraud"] = y_train.values
    
    val_out = X_val.copy()
    val_out["is_fraud"] = y_val.values
    
    test_out = X_test.copy()
    test_out["is_fraud"] = y_test.values
    
    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = {
        "train": output_dir / "train.parquet",
        "val": output_dir / "val.parquet",
        "test": output_dir / "test.parquet",
        "encoder": output_dir / "target_encoder.joblib",
        "features": output_dir / "feature_columns.joblib"
    }
    
    train_out.to_parquet(paths["train"], index=False)
    val_out.to_parquet(paths["val"], index=False)
    test_out.to_parquet(paths["test"], index=False)
    joblib.dump(target_encoder, paths["encoder"])
    joblib.dump(feature_cols, paths["features"])
    
    logger.info(f"Saved train: {paths['train']}")
    logger.info(f"Saved val: {paths['val']}")
    logger.info(f"Saved test: {paths['test']}")
    logger.info(f"Saved encoder: {paths['encoder']}")
    
    return paths


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run feature engineering")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    args = parser.parse_args()
    
    run_feature_engineering(args.input, args.output_dir, args.train_ratio, args.val_ratio)