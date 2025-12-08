"""Production training pipeline - single model."""

import argparse
from pathlib import Path

from src.utils import Config, load_config, get_logger
from src.features.engineering import run_feature_engineering
from src.models.train import (
    load_split_data,
    train_model,
    retrain_on_train_val,
    final_evaluation,
    save_model
)

logger = get_logger(__name__)


def run_pipeline(config: Config) -> None:
    """Run production training pipeline."""
    
    logger.info("=" * 60)
    logger.info("PRODUCTION TRAINING PIPELINE")
    logger.info("=" * 60)
    
    # Step 1: Feature Engineering
    logger.info("Step 1: Feature Engineering")
    input_path = config.paths.processed_data / config.generation.output_filename
    
    paths = run_feature_engineering(
        input_path=input_path,
        output_dir=config.paths.processed_data,
        train_ratio=config.split.train_ratio,
        val_ratio=config.split.val_ratio
    )
    
    # Step 2: Load Data
    logger.info("Step 2: Loading Data")
    X_train, y_train, X_val, y_val, X_test, y_test = load_split_data(
        paths["train"], paths["val"], paths["test"]
    )
    
    # Step 3: Train on Train, Validate on Val
    logger.info("Step 3: Training Model")
    model, val_metrics = train_model(X_train, y_train, X_val, y_val, config)
    logger.info(f"Validation PR-AUC: {val_metrics['pr_auc']:.4f}")
    
    # Step 4: Retrain on Train + Val
    logger.info("Step 4: Retraining on Train + Validation")
    final_model = retrain_on_train_val(X_train, y_train, X_val, y_val, config)
    
    # Step 5: Final Evaluation on Test
    logger.info("Step 5: Final Evaluation on Test")
    test_metrics = final_evaluation(final_model, X_test, y_test, config)
    
    # Step 6: Save Model
    logger.info("Step 6: Saving Model")
    model_path = config.paths.models / "model.joblib"
    save_model(final_model, model_path)
    
    # Summary
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Validation PR-AUC: {val_metrics['pr_auc']:.4f}")
    logger.info(f"Test PR-AUC: {test_metrics['pr_auc']:.4f}")
    logger.info(f"Model: {model_path}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("src/config/config.yaml"))
    args = parser.parse_args()
    
    config = load_config(args.config)
    run_pipeline(config)


if __name__ == "__main__":
    main()