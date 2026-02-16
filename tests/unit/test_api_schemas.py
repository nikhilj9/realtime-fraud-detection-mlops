import pytest
from pydantic import ValidationError

from src.api.schemas.transaction import TransactionRequest


def test_transaction_request_valid(sample_transaction_payload):
    """Test that a valid payload is accepted."""
    request = TransactionRequest(**sample_transaction_payload)
    assert request.amount == 5000.0
    assert request.card_tier == "Gold"

def test_transaction_request_invalid_tier(sample_transaction_payload):
    """Test that invalid categorical values are rejected."""
    payload = sample_transaction_payload.copy()
    payload["card_tier"] = "InvalidTier"  # Should be Classic, Gold, etc.

    with pytest.raises(ValidationError) as exc:
        TransactionRequest(**payload)

    # Verify the error message mentions the field
    assert "card_tier" in str(exc.value)

def test_transaction_request_negative_amount(sample_transaction_payload):
    """Test that negative amounts are rejected."""
    payload = sample_transaction_payload.copy()
    payload["amount"] = -100.0

    with pytest.raises(ValidationError):
        TransactionRequest(**payload)

def test_transaction_request_missing_field(sample_transaction_payload):
    """Test that missing required fields cause error."""
    payload = sample_transaction_payload.copy()
    del payload["V1"]

    with pytest.raises(ValidationError):
        TransactionRequest(**payload)
