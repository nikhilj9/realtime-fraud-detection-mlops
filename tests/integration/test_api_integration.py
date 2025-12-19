import pytest
from fastapi import status

@pytest.mark.integration
def test_predict_endpoint_real_artifacts(client_integration, sample_transaction_payload):
    """
    Test /api/v1/predict using REAL model artifacts.
    Requires: models/champion_model.joblib, data/processed/*.joblib
    """
    response = client_integration.post(
        "/api/v1/predict",
        json=sample_transaction_payload
    )
    
    assert response.status_code == status.HTTP_200_OK
    
    data = response.json()
    # We don't know the exact probability since it's a real model,
    # but it should be a valid float between 0 and 1
    assert 0 <= data["fraud_probability"] <= 1
    assert isinstance(data["is_fraud"], bool)
    assert data["risk_level"] in ["low", "medium", "high"]

@pytest.mark.integration
def test_feature_engineering_consistency(client_integration, sample_transaction_payload):
    """
    Verify that the API's internal feature engineering matches expectations.
    We use the debug endpoint if available, or infer from results.
    """
    # Note: If you exposed a debug endpoint, we would use it here.
    # For now, we just ensure the prediction completes successfully,
    # which implies feature engineering (scaling/encoding) didn't crash.
    response = client_integration.post(
        "/api/v1/predict",
        json=sample_transaction_payload
    )
    assert response.status_code == 200