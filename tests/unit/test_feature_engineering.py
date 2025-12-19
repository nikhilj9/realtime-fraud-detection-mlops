# test_feature_engineering.py
"""Tests for feature engineering module."""

import pytest
import pandas as pd
import joblib

from src.features.engineering import (
    TargetEncoder,
    time_based_split,
    add_time_features,
    add_amount_features,
    add_interaction_features,
    engineer_features,
    get_feature_columns,
    run_feature_engineering,
)


class TestTargetEncoder:
    """Tests for TargetEncoder transformer."""

    def test_input_not_mutated(self, small_fraud_df):
        """Verify transform does not modify input DataFrame."""
        original = small_fraud_df.copy()
        encoder = TargetEncoder(column="merchant_category")
        encoder.fit(small_fraud_df, small_fraud_df["is_fraud"])
        encoder.transform(small_fraud_df)
        pd.testing.assert_frame_equal(small_fraud_df, original, obj="Input DataFrame")

    def test_unknown_category_uses_global_mean(self, small_fraud_df):
        """Verify unknown categories fall back to global mean."""
        encoder = TargetEncoder(column="merchant_category")
        encoder.fit(small_fraud_df, small_fraud_df["is_fraud"])
        new_df = pd.DataFrame({"merchant_category": ["UnknownCategory"]})
        result = encoder.transform(new_df)
        expected = small_fraud_df["is_fraud"].mean()
        actual = result["merchant_category_encoded"].iloc[0]
        assert actual == pytest.approx(expected), "Unknown category should use global mean"

    def test_encoder_persistence(self, small_fraud_df, tmp_path):
        """Verify encoder can be saved and loaded with identical behavior."""
        encoder = TargetEncoder(column="merchant_category")
        encoder.fit(small_fraud_df, small_fraud_df["is_fraud"])
        path = tmp_path / "encoder.joblib"
        joblib.dump(encoder, path)
        loaded = joblib.load(path)
        result_orig = encoder.transform(small_fraud_df)
        result_loaded = loaded.transform(small_fraud_df)
        pd.testing.assert_frame_equal(result_orig, result_loaded, obj="Loaded encoder output")

    def test_smoothing_math_logic(self):
        """
        Verify smoothing formula manually: (n * mean + m * global) / (n + m).
        This ensures the mathematical logic hasn't been broken during refactoring.
        """
        # Create minimal controlled data
        df = pd.DataFrame({"category": ["A", "A", "B"]})
        y = pd.Series([0, 0, 1])  # Global mean = 1/3 (0.333...)
        
        # Initialize with specific smoothing factor
        smoothing_factor = 10
        encoder = TargetEncoder(column="category", smoothing=smoothing_factor)
        encoder.fit(df, y)
        result = encoder.transform(df)
        
        global_mean = 1/3
        
        # Manual Calculation for Category 'A' (n=2, mean=0):
        # (2 * 0.0 + 10 * 0.333) / (2 + 10) = 3.33 / 12 ≈ 0.2777
        numerator = (2 * 0.0) + (smoothing_factor * global_mean)
        denominator = 2 + smoothing_factor
        expected_a = numerator / denominator
        
        actual_a = result["category_encoded"].iloc[0]
        
        # Use approx for floating point comparison
        assert actual_a == pytest.approx(expected_a), "Smoothing formula calculation incorrect"


class TestTimeSplit:
    """Tests for time-based train/val/test split."""

    def test_no_time_leakage(self, sample_fraud_df):
        """Verify train timestamps are strictly before val which are before test."""
        train, val, test = time_based_split(sample_fraud_df)
        assert train["timestamp"].max() <= val["timestamp"].min(), \
            "Train max timestamp must be <= val min timestamp"
        assert val["timestamp"].max() <= test["timestamp"].min(), \
            "Val max timestamp must be <= test min timestamp"

    def test_no_data_loss(self, sample_fraud_df):
        """Verify total rows after split equals original."""
        train, val, test = time_based_split(sample_fraud_df)
        total = len(train) + len(val) + len(test)
        assert total == len(sample_fraud_df), f"Expected {len(sample_fraud_df)} rows, got {total}"

    def test_splits_are_sorted(self, sample_fraud_df):
        """Verify each split is sorted by timestamp."""
        train, val, test = time_based_split(sample_fraud_df)
        for name, df in [("train", train), ("val", val), ("test", test)]:
            is_sorted = df["timestamp"].is_monotonic_increasing
            assert is_sorted, f"{name} split must be sorted by timestamp"


class TestFeatureEngineering:
    """Tests for feature engineering functions."""

    def test_engineer_features_no_mutation(self, small_fraud_df):
        """Verify engineer_features does not modify input."""
        original = small_fraud_df.copy()
        engineer_features(small_fraud_df, fit_encoder=True, y=small_fraud_df["is_fraud"])
        pd.testing.assert_frame_equal(small_fraud_df, original, obj="Input DataFrame")

    def test_time_features_domain_constraints(self, small_fraud_df):
        """Verify time features are within valid domain ranges."""
        df = small_fraud_df.copy()
        add_time_features(df)
        assert df["hour"].between(0, 23).all(), "Hour must be in [0, 23]"
        assert df["day_of_week"].between(0, 6).all(), "Day of week must be in [0, 6]"
        assert df["is_night"].isin([0, 1]).all(), "is_night must be binary"
        assert df["is_weekend"].isin([0, 1]).all(), "is_weekend must be binary"

    def test_log_amount_non_negative(self, small_fraud_df):
        """Verify log_amount is non-negative for valid inputs including zero."""
        df = small_fraud_df.copy()
        add_amount_features(df)
        assert (df["log_amount"] >= 0).all(), "log_amount must be >= 0 for non-negative amounts"
        zero_amount_idx = df[df["amount_inr"] == 0].index
        assert (df.loc[zero_amount_idx, "log_amount"] == 0).all(), "log1p(0) must equal 0"

    def test_interaction_features_binary(self, small_fraud_df):
        """Verify interaction features are binary values."""
        df = small_fraud_df.copy()
        add_time_features(df)
        add_amount_features(df)
        add_interaction_features(df)
        assert df["intl_online"].isin([0, 1]).all(), "intl_online must be binary"
        assert df["night_high_amount"].isin([0, 1]).all(), "night_high_amount must be binary"

    def test_feature_columns_excludes_metadata(self, sample_fraud_df):
        """Verify metadata and target columns are excluded from features."""
        df, _ = engineer_features(sample_fraud_df, fit_encoder=True, y=sample_fraud_df["is_fraud"])
        feature_cols = get_feature_columns(df)
        excluded = ["transaction_id", "user_id", "merchant_id", "timestamp", "is_fraud"]
        for col in excluded:
            assert col not in feature_cols, f"{col} should be excluded from features"

    def test_run_pipeline_creates_usable_artifacts(self, sample_fraud_df, tmp_path):
        """Verify pipeline outputs are loadable and usable for modeling."""
        input_path = tmp_path / "input.parquet"
        sample_fraud_df.to_parquet(input_path)
        output_dir = tmp_path / "output"
        paths = run_feature_engineering(input_path, output_dir)
        train_df = pd.read_parquet(paths["train"])
        encoder = joblib.load(paths["encoder"])
        feature_cols = joblib.load(paths["features"])
        assert len(train_df) > 0, "Train set should not be empty"
        assert all(c in train_df.columns for c in feature_cols), "All features must exist in train"
        assert hasattr(encoder, "transform"), "Encoder must have transform method"

    def test_pipeline_latency_benchmark(self, sample_fraud_df):
        """
        Benchmark feature engineering performance.
        Must process 100 rows in under 50ms to ensure scalability.
        """
        import time
        
        # Warmup (optional, helps exclude import times)
        _ = engineer_features(sample_fraud_df.copy(), fit_encoder=True, y=sample_fraud_df["is_fraud"])
        
        start_time = time.time()
        engineer_features(sample_fraud_df, fit_encoder=True, y=sample_fraud_df["is_fraud"])
        duration = time.time() - start_time
        
        # Threshold: 0.05s (50ms). 
        # If 100 rows take >50ms, 1 million rows will take >8 minutes (too slow for many SLAs).
        assert duration < 0.05, f"Pipeline too slow! Took {duration:.4f}s for 100 rows."