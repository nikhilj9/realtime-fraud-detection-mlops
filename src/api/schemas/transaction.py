"""Request and response schemas for fraud prediction API."""

from typing import Literal

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    """
    Transaction data for fraud prediction.

    Must include all features the model was trained on.
    """

    # === Transaction basics ===
    amount: float = Field(..., gt=0, description="Transaction amount")
    time: float = Field(..., ge=0, description="Seconds since first transaction")

    # === Card information ===
    card_tier: Literal["Classic", "Gold", "Platinum", "Signature"] = Field(
    ...,
    description="Card tier: Classic, Gold, Platinum, or Signature")

    credit_limit: float = Field(..., gt=0, description="Customer credit limit")
    card_age: int = Field(..., ge=0, description="Card age in days")

    # === Transaction context ===
    transaction_channel: Literal["Online", "POS"] = Field(
        ...,
        description="Channel: Online or POS"
    )
    is_international: int = Field(..., ge=0, le=1, description="1 if international, 0 otherwise")
    is_recurring: int = Field(..., ge=0, le=1, description="1 if recurring payment, 0 otherwise")

    # === For target encoding ===
    merchant_category: str | None = Field(
        default=None,
        description="Merchant category for encoding (e.g., 'grocery', 'electronics')"
    )

    # === PCA components (V1-V28) ===
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

    class Config:
        json_schema_extra = {
            "example": {
                "amount": 5000.0,
                "time": 45623.0,
                "card_tier": "Gold",
                "credit_limit": 100000.0,
                "card_age": 365,
                "transaction_channel": "Online",
                "is_international": 0,
                "is_recurring": 0,
                "merchant_category": "electronics",
                "V1": -1.359807,
                "V2": -0.072781,
                "V3": 2.536347,
                "V4": 1.378155,
                "V5": -0.338321,
                "V6": 0.462388,
                "V7": 0.239599,
                "V8": 0.098698,
                "V9": 0.363787,
                "V10": 0.090794,
                "V11": -0.551600,
                "V12": -0.617801,
                "V13": -0.991390,
                "V14": -0.311169,
                "V15": 1.468177,
                "V16": -0.470401,
                "V17": 0.207971,
                "V18": 0.025791,
                "V19": 0.403993,
                "V20": 0.251412,
                "V21": -0.018307,
                "V22": 0.277838,
                "V23": -0.110474,
                "V24": 0.066928,
                "V25": 0.128539,
                "V26": -0.189115,
                "V27": 0.133558,
                "V28": -0.021053,
            }
        }


class PredictionResponse(BaseModel):
    """Fraud prediction result."""

    is_fraud: bool = Field(..., description="True if predicted as fraudulent")
    fraud_probability: float = Field(..., ge=0, le=1, description="Probability of fraud (0-1)")
    risk_level: str = Field(..., description="Risk category: low, medium, or high")

    class Config:
        json_schema_extra = {
            "example": {
                "is_fraud": False,
                "fraud_probability": 0.12,
                "risk_level": "low"
            }
        }
