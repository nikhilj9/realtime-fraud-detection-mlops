"""
Feature engineering with S3/MinIO support.
Standard: Uses Object Storage for Atomic Writes.
"""
import argparse
import gc
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import psutil

from src.features.encoders import TargetEncoder
from src.utils import get_logger

logger = get_logger(__name__)

# =============================================================================
# TIME-BASED SPLIT (3-WAY)
# =============================================================================

def log_memory(label=""):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024**2
    logger.info(f"MEMORY [{label}]: {mem_mb:.1f} MB")

# Downcast map: reduces float64→float32, int64→int8/16, strings→category
dtype_map = {
    'user_id': 'category',
    'merchant_id': 'category',

    'is_fraud': 'int8',
    'is_international': 'int8',
    'is_recurring': 'int8',
    'card_age': 'int16',

    **{f'V{i}': 'float32' for i in range(1, 29)},

    'amount_inr': 'float32',
    'credit_limit': 'float32',

    'card_network': 'category',
    'card_issuer': 'category',
    'card_tier': 'category',
    'merchant_category': 'category',
    'merchant_city': 'category',
    'merchant_state': 'category',
    'transaction_channel': 'category',
    'entry_mode': 'category',
    'merchant_category_encoded': 'float32',
}

def time_based_split(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    timestamp_col: str = "timestamp"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train/validation/test by time."""

    # Sort in-place to avoid creating a copy
    df.sort_values(timestamp_col, inplace=True)
    df.reset_index(drop=True, inplace=True)

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    # Only copy the smallest splits; train can be a slice since we del df right after
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


# =============================================================================
# FEATURE ENGINEERING LOGIC
# =============================================================================

def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Kaggle Credit Card dataset to expected schema. Modifies in-place."""

    if "Class" in df.columns:
        logger.info("Detected Kaggle Credit Card dataset - normalizing schema")
        df["is_fraud"] = df["Class"]
        df["amount_inr"] = df["Amount"]
        df["timestamp"] = pd.to_datetime("2024-01-01") + pd.to_timedelta(df["Time"], unit='ns')

    return df

def add_time_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> None:
    ts = pd.to_datetime(df[timestamp_col]) if timestamp_col in df.columns else pd.to_datetime("2024-01-01")
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["is_night"] = ts.dt.hour.isin([0, 1, 2, 3, 4, 5]).astype(int)
    df["is_weekend"] = ts.dt.dayofweek.isin([5, 6]).astype(int)

def add_amount_features(df: pd.DataFrame, amount_col: str = "amount_inr") -> None:
    df["log_amount"] = np.log1p(df[amount_col])

def add_interaction_features(df: pd.DataFrame) -> None:
    if "is_international" in df.columns and "transaction_channel" in df.columns:
        df["intl_online"] = ((df["is_international"] == 1) & (df["transaction_channel"] == "Online")).astype(int)

    if "is_night" in df.columns and "log_amount" in df.columns:
        df["night_high_amount"] = ((df["is_night"] == 1) & (df["log_amount"] > df["log_amount"].quantile(0.9))).astype(int)

def engineer_features(
    df: pd.DataFrame,
    target_encoder: TargetEncoder = None,
    fit_encoder: bool = False,
    y: pd.Series = None
) -> tuple[pd.DataFrame, TargetEncoder]:
    """Add engineered features. Modifies df in-place since caller reassigns result."""

    add_time_features(df)
    add_amount_features(df)
    add_interaction_features(df)

    if fit_encoder and y is not None and "merchant_category" in df.columns:
        target_encoder = TargetEncoder(column="merchant_category")
        target_encoder.fit(df, y)

    if target_encoder:
        df = target_encoder.transform(df)

    return df, target_encoder

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = [
        "transaction_id", "user_id", "merchant_id", "timestamp",
        "card_network", "card_issuer", "entry_mode",
        "merchant_category", "merchant_city", "merchant_state",
        "is_fraud",
        "Time", "Class", "Amount"
    ]
    return [c for c in df.columns if c not in exclude]

def prepare_modeling_data(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    # Multi-column selection already creates a new DataFrame
    X = df[feature_cols]
    # Single column is a view, so .copy() is needed to safely free source df later
    y = df["is_fraud"].copy()
    return X, y


# =============================================================================
# MAIN PIPELINE (S3 & LOCAL COMPATIBLE)
# =============================================================================

def run_feature_engineering(
    input_path: Any,
    output_dir: str,
    storage_options: dict[str, Any] | None = None,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2
) -> dict[str, str]:
    """
    Run feature engineering pipeline.
    Expects LOCAL paths only (S3 download handled upstream).
    """

    logger.info("=" * 50)
    logger.info("Starting Feature Engineering")
    logger.info("=" * 50)

    input_path = Path(input_path)
    output_dir = Path(output_dir)

    # 1. Load data
    try:
        if input_path.suffix == ".csv":
            logger.info(f"Loading CSV from {input_path}")
            df = pd.read_csv(input_path)
        else:
            logger.info(f"Loading Parquet from {input_path}")
            df = pd.read_parquet(input_path)

        logger.info(f"Successfully loaded data. Shape: {df.shape}")
        log_memory("after load")

        # Downcast dtypes to reduce memory (float64→float32, int64→int8, etc.)
        applicable = {col: dtype for col, dtype in dtype_map.items() if col in df.columns}
        if applicable:
            df = df.astype(applicable)
            logger.info(f"Downcasted {len(applicable)} columns to save memory")
            log_memory("after downcast")

    except Exception as e:
        logger.error(f"Failed to load data from {input_path}: {e}")
        raise

    # Normalize schema (adds is_fraud, amount_inr, timestamp if Kaggle format)
    df = normalize_schema(df)

    # 2. Split by time
    train_df, val_df, test_df = time_based_split(df, train_ratio, val_ratio)

    # Free original df — split copies are independent now
    del df
    gc.collect()
    log_memory("after split")

    # 3. Engineer features
    y_train = train_df["is_fraud"]
    train_df, target_encoder = engineer_features(train_df, fit_encoder=True, y=y_train)
    val_df, _ = engineer_features(val_df, target_encoder=target_encoder)
    test_df, _ = engineer_features(test_df, target_encoder=target_encoder)

    log_memory("after FE")

    # 4. Extract modeling columns
    feature_cols = get_feature_columns(train_df)

    X_train, y_train = prepare_modeling_data(train_df, feature_cols)
    X_val, y_val = prepare_modeling_data(val_df, feature_cols)
    X_test, y_test = prepare_modeling_data(test_df, feature_cols)

    # Free split dfs — X/y extracted, originals no longer needed
    del train_df, val_df, test_df
    gc.collect()
    log_memory("after cleanup")

    # 5. Build output (modify X in-place since it's not used after saving)
    X_train["is_fraud"] = y_train.values
    X_val["is_fraud"] = y_val.values
    X_test["is_fraud"] = y_test.values

    # 6. Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "train": str(output_dir / "train.parquet"),
        "val": str(output_dir / "val.parquet"),
        "test": str(output_dir / "test.parquet"),
        "encoder": str(output_dir / "target_encoder.joblib"),
        "features": str(output_dir / "feature_columns.joblib")
    }

    logger.info(f"Writing to: {output_dir}")

    X_train.to_parquet(paths["train"], index=False)
    X_val.to_parquet(paths["val"], index=False)
    X_test.to_parquet(paths["test"], index=False)

    logger.info(f"train_out: {paths['train']}")
    logger.info(f"val_out: {paths['val']}")
    logger.info(f"test_out: {paths['test']}")

    joblib.dump(target_encoder, paths["encoder"])
    joblib.dump(feature_cols, paths["features"])

    return paths


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run feature engineering")

    parser.add_argument("--input", type=str, required=True, help="Input Parquet path (local or s3://)")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Output directory (local or s3://)")
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--s3-endpoint", type=str, default=None, help="S3 Endpoint URL (for MinIO)")

    args = parser.parse_args()

    storage_opts = None
    if args.input.startswith("s3://") or args.output_dir.startswith("s3://"):
        storage_opts = {
            "key": os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"),
            "secret": os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin"),
        }
        if args.s3_endpoint:
             storage_opts["client_kwargs"] = {"endpoint_url": args.s3_endpoint}

    run_feature_engineering(
        input_path=args.input,
        output_dir=args.output_dir,
        storage_options=storage_opts,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )
