# tests/integration/test_api_integration.py
"""Integration tests for API with real (fixture-generated) artifacts."""

import pytest
from fastapi import status


@pytest.mark.integration
def test_predict_endpoint_real_artifacts(client_with_real_artifacts, sample_transaction_payload):
    """
    Test /api/v1/predict using REAL (fixture-generated) model artifacts.

    Uses:
    - Real trained LogisticRegression model
    - Real fitted TargetEncoder
    - Real feature columns list

    Tests the full prediction pipeline with real objects.
    """
    response = client_with_real_artifacts.post(
        "/api/v1/predict",
        json=sample_transaction_payload
    )

    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    # Validate response structure and value ranges
    assert 0 <= data["fraud_probability"] <= 1
    assert isinstance(data["is_fraud"], bool)
    assert data["risk_level"] in ["low", "medium", "high"]


@pytest.mark.integration
def test_feature_engineering_consistency(client_with_real_artifacts, sample_transaction_payload):
    """
    Verify that the API's internal feature engineering works correctly.

    The prediction completing successfully implies:
    - Feature engineering (time features, amount features) worked
    - Encoder transformation worked
    - Model prediction worked
    """
    response = client_with_real_artifacts.post(
        "/api/v1/predict",
        json=sample_transaction_payload
    )
    assert response.status_code == status.HTTP_200_OK


@pytest.mark.integration
def test_prediction_with_different_amounts(client_with_real_artifacts, sample_transaction_payload):
    """Test predictions with various transaction amounts."""
    test_amounts = [100.0, 1000.0, 10000.0, 50000.0]

    for amount in test_amounts:
        payload = sample_transaction_payload.copy()
        payload["amount"] = amount

        response = client_with_real_artifacts.post(
            "/api/v1/predict",
            json=payload
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert 0 <= data["fraud_probability"] <= 1, f"Invalid probability for amount={amount}"


@pytest.mark.integration
def test_prediction_with_international_transaction(client_with_real_artifacts, sample_transaction_payload):
    """Test prediction for international transactions."""
    payload = sample_transaction_payload.copy()
    payload["is_international"] = 1

    response = client_with_real_artifacts.post(
        "/api/v1/predict",
        json=payload
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert 0 <= data["fraud_probability"] <= 1
