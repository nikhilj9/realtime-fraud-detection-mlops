"""Configuration management with Pydantic validation."""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class PathConfig(BaseModel):
    """File paths configuration."""
    raw_data: Path
    processed_data: Path = Path("data/processed")
    models: Path = Path("models")
    logs: Path = Path("logs")
    
    @field_validator("raw_data")
    @classmethod
    def check_raw_exists(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Raw data not found: {v}")
        return v
    
    def ensure_dirs(self) -> None:
        """Create output directories."""
        self.processed_data.mkdir(parents=True, exist_ok=True)
        self.models.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)


class SplitConfig(BaseModel):
    """Data split configuration."""
    train_ratio: float = Field(default=0.6, gt=0, lt=1)
    val_ratio: float = Field(default=0.2, gt=0, lt=1)
    test_ratio: float = Field(default=0.2, gt=0, lt=1)
    
    @model_validator(mode="after")
    def validate_ratios(self) -> "SplitConfig":
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        return self


class GenerationParams(BaseModel):
    """Parameters for synthetic data generation."""
    random_seed: int = Field(default=42, ge=0)
    exchange_rate: float = Field(default=88.0, gt=0)
    n_users: int = Field(default=10000, gt=0)
    n_merchants: int = Field(default=5000, gt=0)
    start_date: str = Field(default="2023-09-02")
    output_filename: str = Field(default="transactions_enriched.parquet")


class DistributionConfig(BaseModel):
    """Probability distribution for categorical assignment."""
    low_amount: Dict[str, float]
    mid_amount: Dict[str, float]
    high_amount: Dict[str, float]
    
    @model_validator(mode="after")
    def validate_probabilities(self) -> "DistributionConfig":
        for name in ["low_amount", "mid_amount", "high_amount"]:
            dist = getattr(self, name)
            total = sum(dist.values())
            if abs(total - 1.0) > 0.001:
                raise ValueError(f"{name} probabilities sum to {total}, expected 1.0")
        return self


class FeatureConfig(BaseModel):
    """Feature engineering parameters."""
    amount_log_transform: bool = True
    time_features: bool = True
    target_encoding_smoothing: float = Field(default=10.0, ge=0)


class ModelParams(BaseModel):
    """Model hyperparameters."""
    model_type: str = Field(default="xgboost", pattern="^(xgboost|lightgbm|random_forest)$")
    random_state: int = Field(default=42, ge=0)
    n_estimators: int = Field(default=100, gt=0)
    max_depth: int = Field(default=6, gt=0)
    learning_rate: float = Field(default=0.1, gt=0, le=1)
    subsample: float = Field(default=0.8, gt=0, le=1)
    colsample_bytree: float = Field(default=0.8, gt=0, le=1)
    smote_ratio: float = Field(default=0.1, gt=0, le=1)
    
    def to_xgb_params(self) -> Dict[str, Any]:
        """Return XGBoost params dict."""
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "random_state": self.random_state,
            "n_jobs": -1,
            "verbosity": 0
        }


class MLflowConfig(BaseSettings):
    """MLflow tracking configuration."""
    experiment_name: str = Field(default="fraud-detection")
    tracking_uri: str = Field(
        default="mlruns", 
        validation_alias="MLFLOW_TRACKING_URI"
    )
    log_models: bool = True
    # This inner class tells Pydantic to read from the .env file
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore" # Ignores other variables in .env not defined here
    )


class Config(BaseModel):
    """Complete pipeline configuration."""
    paths: PathConfig
    split: SplitConfig = SplitConfig()
    generation: GenerationParams = GenerationParams()
    distributions: Optional[Dict[str, DistributionConfig]] = None
    features: FeatureConfig = FeatureConfig()
    model: ModelParams = ModelParams()
    mlflow: MLflowConfig = MLflowConfig()


def load_config(config_path: Path) -> Config:
    """Load and validate config from YAML file."""
    from src.utils.exceptions import ConfigurationError
    
    if not config_path.exists():
        raise ConfigurationError(f"Config file not found: {config_path}")
    
    try:
        with open(config_path) as f:
            raw = yaml.safe_load(f)
        
        config = Config(**raw)
        config.paths.ensure_dirs()
        
        return config
    
    except Exception as e:
        raise ConfigurationError(f"Invalid config: {e}") from e