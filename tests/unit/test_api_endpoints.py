import pytest
from fastapi import status

@pytest.mark.unit
def test_health_check(client_with_mocks):
    """Test /health endpoint. Does not need model."""
    response = client_with_mocks.get("/health")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "healthy"}

@pytest.mark.unit
def test_predict_endpoint_mocked(client_with_mocks, sample_transaction_payload):
    """
    Test /api/v1/predict using MOCKED model.
    This proves the API code works (receiving, parsing, returning)
    without needing the actual heavy model file.
    """
    response = client_with_mocks.post(
        "/api/v1/predict",
        json=sample_transaction_payload
    )
    
    # 1. Check HTTP Status
    assert response.status_code == status.HTTP_200_OK
    
    # 2. Check Response Structure
    data = response.json()
    assert "is_fraud" in data
    assert "fraud_probability" in data
    assert "risk_level" in data
    
    # 3. Validate Logic (Our mock returns 0.2 probability)
    # Since 0.2 < 0.5, is_fraud should be False
    assert data["fraud_probability"] == 0.2
    assert data["is_fraud"] is False
    assert data["risk_level"] == "low"

@pytest.mark.unit
def test_predict_endpoint_invalid_input(client_with_mocks):
    """Test that API returns 422 for bad input."""
    response = client_with_mocks.post(
        "/api/v1/predict",
        json={"amount": "invalid"} # Missing all other fields
    )
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT