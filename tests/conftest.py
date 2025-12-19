# tests/conftest.py (UPDATED - add to existing file)
"""Shared fixtures for all test modules."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

from fastapi.testclient import TestClient
from src.api.app import app

# =============================================================================
# COMMON FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Session-scoped temporary directory for test artifacts."""
    return tmp_path_factory.mktemp("test_data")


# =============================================================================
# FEATURE ENGINEERING FIXTURES
# =============================================================================

@pytest.fixture
def sample_fraud_df():
    """Realistic fraud detection dataset with seed=42."""
    np.random.seed(42)
    n = 100
    timestamps = pd.date_range("2024-01-01", periods=n, freq="H")
    
    return pd.DataFrame({
        "transaction_id": [f"TXN_{i:05d}" for i in range(n)],
        "user_id": [f"USER_{i % 20:03d}" for i in range(n)],
        "merchant_id": [f"MERCH_{i % 10:03d}" for i in range(n)],
        "timestamp": timestamps,
        "amount_inr": np.random.exponential(1000, n),
        "is_international": np.random.choice([0, 1], n, p=[0.8, 0.2]),
        "transaction_channel": np.random.choice(["Online", "POS", "ATM"], n),
        "merchant_category": np.random.choice(["Retail", "Food", "Travel", "Electronics"], n),
        "card_network": np.random.choice(["Visa", "Mastercard"], n),
        "card_issuer": np.random.choice(["Bank_A", "Bank_B"], n),
        "entry_mode": np.random.choice(["Chip", "Swipe"], n),
        "merchant_city": np.random.choice(["Mumbai", "Delhi"], n),
        "merchant_state": np.random.choice(["MH", "DL"], n),
        "is_fraud": np.random.choice([0, 1], n, p=[0.95, 0.05]),
    })


@pytest.fixture
def small_fraud_df():
    """Minimal dataset for edge case testing."""
    return pd.DataFrame({
        "transaction_id": ["TXN_001", "TXN_002", "TXN_003", "TXN_004", "TXN_005"],
        "user_id": ["U1", "U2", "U3", "U4", "U5"],
        "merchant_id": ["M1", "M1", "M2", "M2", "M3"],
        "timestamp": pd.to_datetime([
            "2024-01-01 02:00:00", "2024-01-01 14:00:00", "2024-01-02 03:00:00",
            "2024-01-02 15:00:00", "2024-01-03 04:00:00",
        ]),
        "amount_inr": [100.0, 500.0, 0.0, 10000.0, 50.0],
        "is_international": [0, 1, 0, 1, 0],
        "transaction_channel": ["Online", "POS", "Online", "Online", "ATM"],
        "merchant_category": ["Retail", "Food", "Retail", "Travel", "Food"],
        "card_network": ["Visa", "Mastercard", "Visa", "Visa", "Mastercard"],
        "card_issuer": ["Bank_A", "Bank_B", "Bank_A", "Bank_A", "Bank_B"],
        "entry_mode": ["Chip", "Swipe", "Chip", "Chip", "Swipe"],
        "merchant_city": ["Mumbai", "Delhi", "Mumbai", "Delhi", "Mumbai"],
        "merchant_state": ["MH", "DL", "MH", "DL", "MH"],
        "is_fraud": [0, 0, 1, 0, 1],
    })


# =============================================================================
# MODEL TRAINING FIXTURES
# =============================================================================

@pytest.fixture
def sample_train_df():
    """Realistic training data with engineered features."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "amount_inr": np.random.exponential(1000, n),
        "log_amount": np.log1p(np.random.exponential(1000, n)),
        "hour": np.random.randint(0, 24, n),
        "day_of_week": np.random.randint(0, 7, n),
        "is_night": np.random.choice([0, 1], n),
        "is_weekend": np.random.choice([0, 1], n),
        "is_international": np.random.choice([0, 1], n),
        "transaction_channel": np.random.choice(["Online", "POS", "ATM"], n),
        "merchant_category_encoded": np.random.uniform(0, 0.1, n),
        "intl_online": np.random.choice([0, 1], n),
        "night_high_amount": np.random.choice([0, 1], n),
        "is_fraud": np.random.choice([0, 1], n, p=[0.95, 0.05]),
    })


@pytest.fixture
def sample_val_df(sample_train_df):
    """Validation data with same schema."""
    np.random.seed(43)
    return sample_train_df.sample(frac=0.5, random_state=43).reset_index(drop=True)


@pytest.fixture
def sample_test_df(sample_train_df):
    """Test data with same schema."""
    np.random.seed(44)
    return sample_train_df.sample(frac=0.3, random_state=44).reset_index(drop=True)


@pytest.fixture
def mock_config():
    """Mock Config object with model parameters."""
    config = MagicMock()
    config.model.n_estimators = 10
    config.model.max_depth = 3
    config.model.learning_rate = 0.1
    config.model.subsample = 0.8
    config.model.colsample_bytree = 0.8
    config.model.random_state = 42
    config.mlflow.tracking_uri = "sqlite:///mlruns.db"
    config.mlflow.experiment_name = "test_experiment"
    return config


@pytest.fixture
def parquet_splits(sample_train_df, sample_val_df, sample_test_df, tmp_path):
    """Save splits as parquet files and return paths."""
    train_path = tmp_path / "train.parquet"
    val_path = tmp_path / "val.parquet"
    test_path = tmp_path / "test.parquet"
    sample_train_df.to_parquet(train_path)
    sample_val_df.to_parquet(val_path)
    sample_test_df.to_parquet(test_path)
    return train_path, val_path, test_path


# =============================================================================
# MODEL PREDICTION FIXTURES
# =============================================================================

@pytest.fixture
def numeric_feature_df():
    """Numeric-only DataFrame for prediction testing."""
    np.random.seed(42)
    n = 50
    return pd.DataFrame({
        "log_amount": np.random.uniform(0, 10, n),
        "hour": np.random.randint(0, 24, n),
        "day_of_week": np.random.randint(0, 7, n),
        "is_night": np.random.choice([0, 1], n),
        "is_weekend": np.random.choice([0, 1], n),
        "is_international": np.random.choice([0, 1], n),
        "merchant_category_encoded": np.random.uniform(0, 0.1, n),
    })


@pytest.fixture
def trained_sklearn_model(numeric_feature_df):
    """Real trained sklearn pipeline for testing."""
    np.random.seed(42)
    X = numeric_feature_df
    y = np.random.choice([0, 1], len(X), p=[0.9, 0.1])
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(random_state=42, max_iter=200))
    ])
    pipeline.fit(X, y)
    return pipeline


@pytest.fixture
def saved_model_path(trained_sklearn_model, tmp_path):
    """Save trained model and return path."""
    path = tmp_path / "model.joblib"
    joblib.dump(trained_sklearn_model, path)
    return path


@pytest.fixture
def prediction_input_parquet(numeric_feature_df, tmp_path):
    """Parquet file with features for batch prediction."""
    df = numeric_feature_df.copy()
    df["is_fraud"] = np.random.choice([0, 1], len(df), p=[0.9, 0.1])
    path = tmp_path / "input.parquet"
    df.to_parquet(path)
    return path


# =============================================================================
# MODEL EVALUATION FIXTURES
# =============================================================================

@pytest.fixture
def binary_classification_data():
    """Ground truth and predictions for evaluation testing."""
    np.random.seed(42)
    n = 100
    y_true = np.random.choice([0, 1], n, p=[0.9, 0.1])
    y_prob = np.clip(y_true + np.random.normal(0, 0.3, n), 0, 1)
    y_pred = (y_prob >= 0.5).astype(int)
    return y_true, y_pred, y_prob


@pytest.fixture
def all_negative_data():
    """Edge case: all samples are negative class."""
    n = 50
    y_true = np.zeros(n, dtype=int)
    y_pred = np.zeros(n, dtype=int)
    y_prob = np.random.uniform(0, 0.4, n)
    return y_true, y_pred, y_prob


@pytest.fixture
def model_with_feature_importance(numeric_feature_df):
    """RandomForest model with feature_importances_ attribute."""
    np.random.seed(42)
    X = numeric_feature_df
    y = np.random.choice([0, 1], len(X), p=[0.9, 0.1])
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=10, random_state=42))
    ])
    pipeline.fit(X, y)
    return pipeline


@pytest.fixture
def model_without_feature_importance(numeric_feature_df):
    """LogisticRegression model without feature_importances_."""
    np.random.seed(42)
    X = numeric_feature_df
    y = np.random.choice([0, 1], len(X), p=[0.9, 0.1])
    model = LogisticRegression(random_state=42, max_iter=200)
    model.fit(X, y)
    return model


# =============================================================================
# PIPELINE FIXTURES
# =============================================================================

@pytest.fixture
def pipeline_config(tmp_path):
    """Complete mock config for pipeline testing."""
    config = MagicMock()
    
    config.paths.processed_data = tmp_path / "processed"
    config.paths.models = tmp_path / "models"
    config.paths.processed_data.mkdir(parents=True, exist_ok=True)
    config.paths.models.mkdir(parents=True, exist_ok=True)
    
    config.generation.output_filename = "transactions.parquet"
    config.split.train_ratio = 0.6
    config.split.val_ratio = 0.2
    
    config.model.n_estimators = 10
    config.model.max_depth = 3
    config.model.learning_rate = 0.1
    config.model.subsample = 0.8
    config.model.colsample_bytree = 0.8
    config.model.random_state = 42
    
    config.mlflow.tracking_uri = "sqlite:///mlruns.db"
    config.mlflow.experiment_name = "test_pipeline"
    
    return config


@pytest.fixture
def mock_pipeline_dependencies(sample_train_df, sample_val_df, sample_test_df, tmp_path):
    """Mock return values for pipeline dependencies."""
    train_path = tmp_path / "train.parquet"
    val_path = tmp_path / "val.parquet"
    test_path = tmp_path / "test.parquet"
    
    sample_train_df.to_parquet(train_path)
    sample_val_df.to_parquet(val_path)
    sample_test_df.to_parquet(test_path)
    
    return {"train": train_path, "val": val_path, "test": test_path}


# =============================================================================
# INTEGRATION TEST FIXTURES
# =============================================================================

@pytest.fixture
def e2e_raw_data_path(sample_fraud_df, tmp_path):
    """Raw transaction data saved as parquet for E2E tests."""
    path = tmp_path / "raw_transactions.parquet"
    sample_fraud_df.to_parquet(path)
    return path


@pytest.fixture
def e2e_processed_splits(sample_train_df, sample_val_df, sample_test_df, tmp_path):
    """Processed splits for E2E model training tests."""
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = processed_dir / "train.parquet"
    val_path = processed_dir / "val.parquet"
    test_path = processed_dir / "test.parquet"
    
    sample_train_df.to_parquet(train_path)
    sample_val_df.to_parquet(val_path)
    sample_test_df.to_parquet(test_path)
    
    return train_path, val_path, test_path


@pytest.fixture
def mlflow_tracking_uri(tmp_path):
    """Isolated MLflow tracking URI for testing."""
    uri = f"sqlite:///{tmp_path}/mlruns.db"
    return uri

# =============================================================================
# API / FASTAPI FIXTURES
# =============================================================================

@pytest.fixture
def api_client():
    """Client for calling API endpoints."""
    return TestClient(app)

@pytest.fixture
def mock_model_artifact():
    """Mock XGBoost/Sklearn model for unit tests."""
    model = MagicMock()
    # Mock predict_proba to return [0.8, 0.2] (not fraud)
    model.predict_proba.return_value = np.array([[0.8, 0.2]]) 
    return model

@pytest.fixture
def mock_encoder_artifact():
    """Mock TargetEncoder for unit tests."""
    encoder = MagicMock()
    # Mock transform to return the same dataframe
    encoder.transform.side_effect = lambda df: df
    return encoder

@pytest.fixture
def mock_feature_columns():
    """Mock feature columns list."""
    return [
        "amount_inr", "log_amount", "hour", "day_of_week", "is_night", 
        "is_weekend", "intl_online", "night_high_amount", 
        "merchant_category_encoded"
    ] + [f"V{i}" for i in range(1, 29)]

@pytest.fixture
def client_with_mocks(mock_model_artifact, mock_encoder_artifact, mock_feature_columns):
    """
    TestClient that mocks the artifact loading (unit tests).
    This prevents DVC files from being required.
    """
    # Patch joblib.load to return our mocks instead of reading files
    with patch("src.api.app.joblib.load") as mock_load:
        def side_effect(path):
            path_str = str(path)
            if "model" in path_str:
                return mock_model_artifact
            elif "encoder" in path_str:
                return mock_encoder_artifact
            elif "feature_columns" in path_str:
                return mock_feature_columns
            return MagicMock()
            
        mock_load.side_effect = side_effect
        
        # Create client inside the patch context
        with TestClient(app) as client:
            yield client

@pytest.fixture
def client_integration():
    """
    TestClient that uses REAL artifacts (Integration tests).
    Requires DVC files to be present.
    """
    with TestClient(app) as client:
        yield client

@pytest.fixture
def sample_transaction_payload():
    """Valid transaction JSON payload."""
    return {
        "amount": 5000.0,
        "time": 45623.0,
        "card_tier": "Gold",
        "credit_limit": 100000.0,
        "card_age": 365,
        "transaction_channel": "Online",
        "is_international": 0,
        "is_recurring": 0,
        "merchant_category": "electronics",
        "V1": -1.359807, "V2": -0.072781, "V3": 2.536347, "V4": 1.378155,
        "V5": -0.338321, "V6": 0.462388, "V7": 0.239599, "V8": 0.098698,
        "V9": 0.363787, "V10": 0.090794, "V11": -0.551600, "V12": -0.617801,
        "V13": -0.991390, "V14": -0.311169, "V15": 1.468177, "V16": -0.470401,
        "V17": 0.207971, "V18": 0.025791, "V19": 0.403993, "V20": 0.251412,
        "V21": -0.018307, "V22": 0.277838, "V23": -0.110474, "V24": 0.066928,
        "V25": 0.128539, "V26": -0.189115, "V27": 0.133558, "V28": -0.021053
    }