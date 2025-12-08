"""Custom exceptions for fraud detection pipeline."""


class FraudDetectionError(Exception):
    """Base exception for all pipeline errors."""
    pass


class ConfigurationError(FraudDetectionError):
    """Invalid or missing configuration."""
    pass


class DataLoadError(FraudDetectionError):
    """Data loading failed."""
    pass


class DataValidationError(FraudDetectionError):
    """Data validation failed."""
    pass


class DataGenerationError(FraudDetectionError):
    """Data generation failed."""
    pass


class FeatureEngineeringError(FraudDetectionError):
    """Feature engineering failed."""
    pass


class ModelTrainingError(FraudDetectionError):
    """Model training failed."""
    pass


class ModelPredictionError(FraudDetectionError):
    """Model prediction failed."""
    pass