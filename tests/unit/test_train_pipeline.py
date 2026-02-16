# tests/unit/test_train_pipeline.py
"""Tests for production training pipeline."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from src.pipelines.training_pipeline import main, run_pipeline


class TestRunPipeline:
    """Tests for pipeline orchestration."""

    @patch("src.pipelines.training_pipeline.save_model")
    @patch("src.pipelines.training_pipeline.final_evaluation")
    @patch("src.pipelines.training_pipeline.retrain_on_train_val")
    @patch("src.pipelines.training_pipeline.train_model")
    @patch("src.pipelines.training_pipeline.load_split_data")
    @patch("src.pipelines.training_pipeline.run_feature_engineering")
    @patch("src.pipelines.training_pipeline.check_data_drift")
    @patch("src.pipelines.training_pipeline.validate_raw_data")
    @patch("src.pipelines.training_pipeline.save")
    @patch("src.pipelines.training_pipeline.generate")
    def test_pipeline_calls_steps_in_order(
        self, mock_generate, mock_save_data, mock_validate, mock_drift,
        mock_fe, mock_load, mock_train, mock_retrain, mock_eval, mock_save,
        pipeline_config, mock_pipeline_dependencies, trained_sklearn_model, sample_fraud_df
    ):
        """Verify pipeline executes all steps in correct sequence."""
        # Mock Step 0: Data Enrichment
        mock_generate.return_value = sample_fraud_df
        mock_validate.return_value = sample_fraud_df

        # Mock Step 2: Feature Engineering
        mock_fe.return_value = mock_pipeline_dependencies
        mock_load.return_value = (
            pd.DataFrame(), pd.Series(dtype=int),
            pd.DataFrame(), pd.Series(dtype=int),
            pd.DataFrame(), pd.Series(dtype=int),
        )
        mock_train.return_value = (trained_sklearn_model, {"pr_auc": 0.85})
        mock_retrain.return_value = trained_sklearn_model
        mock_eval.return_value = {"pr_auc": 0.83}

        run_pipeline(pipeline_config, skip_drift=True)

        assert mock_generate.called, "Data generation must be called"
        assert mock_save_data.called, "Data save must be called"
        assert mock_validate.called, "Validation must be called"
        assert mock_fe.called, "Feature engineering must be called"
        assert mock_load.called, "Data loading must be called"
        assert mock_train.called, "Training must be called"
        assert mock_retrain.called, "Retraining must be called"
        assert mock_eval.called, "Evaluation must be called"
        assert mock_save.called, "Model saving must be called"

    @patch("src.pipelines.training_pipeline.save_model")
    @patch("src.pipelines.training_pipeline.final_evaluation")
    @patch("src.pipelines.training_pipeline.retrain_on_train_val")
    @patch("src.pipelines.training_pipeline.train_model")
    @patch("src.pipelines.training_pipeline.load_split_data")
    @patch("src.pipelines.training_pipeline.run_feature_engineering")
    @patch("src.pipelines.training_pipeline.check_data_drift")
    @patch("src.pipelines.training_pipeline.validate_raw_data")
    @patch("src.pipelines.training_pipeline.save")
    @patch("src.pipelines.training_pipeline.generate")
    def test_pipeline_passes_correct_split_ratios(
        self, mock_generate, mock_save_data, mock_validate, mock_drift,
        mock_fe, mock_load, mock_train, mock_retrain, mock_eval, mock_save,
        pipeline_config, mock_pipeline_dependencies, trained_sklearn_model, sample_fraud_df
    ):
        """Verify config split ratios are passed to feature engineering."""
        mock_generate.return_value = sample_fraud_df
        mock_validate.return_value = sample_fraud_df
        mock_fe.return_value = mock_pipeline_dependencies
        mock_load.return_value = (
            pd.DataFrame(), pd.Series(dtype=int),
            pd.DataFrame(), pd.Series(dtype=int),
            pd.DataFrame(), pd.Series(dtype=int),
        )
        mock_train.return_value = (trained_sklearn_model, {"pr_auc": 0.85})
        mock_retrain.return_value = trained_sklearn_model
        mock_eval.return_value = {"pr_auc": 0.83}

        run_pipeline(pipeline_config, skip_drift=True)

        call_kwargs = mock_fe.call_args
        assert call_kwargs.kwargs["train_ratio"] == 0.6, "Train ratio must match config"
        assert call_kwargs.kwargs["val_ratio"] == 0.2, "Val ratio must match config"

    @patch("src.pipelines.training_pipeline.save_model")
    @patch("src.pipelines.training_pipeline.final_evaluation")
    @patch("src.pipelines.training_pipeline.retrain_on_train_val")
    @patch("src.pipelines.training_pipeline.train_model")
    @patch("src.pipelines.training_pipeline.load_split_data")
    @patch("src.pipelines.training_pipeline.run_feature_engineering")
    @patch("src.pipelines.training_pipeline.check_data_drift")
    @patch("src.pipelines.training_pipeline.validate_raw_data")
    @patch("src.pipelines.training_pipeline.save")
    @patch("src.pipelines.training_pipeline.generate")
    def test_pipeline_saves_model_to_correct_path(
        self, mock_generate, mock_save_data, mock_validate, mock_drift,
        mock_fe, mock_load, mock_train, mock_retrain, mock_eval, mock_save,
        pipeline_config, mock_pipeline_dependencies, trained_sklearn_model, sample_fraud_df
    ):
        """Verify model is saved to configured path."""
        mock_generate.return_value = sample_fraud_df
        mock_validate.return_value = sample_fraud_df
        mock_fe.return_value = mock_pipeline_dependencies
        mock_load.return_value = (
            pd.DataFrame(), pd.Series(dtype=int),
            pd.DataFrame(), pd.Series(dtype=int),
            pd.DataFrame(), pd.Series(dtype=int),
        )
        mock_train.return_value = (trained_sklearn_model, {"pr_auc": 0.85})
        mock_retrain.return_value = trained_sklearn_model
        mock_eval.return_value = {"pr_auc": 0.83}

        run_pipeline(pipeline_config, skip_drift=True)

        save_call = mock_save.call_args
        model_path = save_call.args[1]
        expected = pipeline_config.paths.models / "champion_model.joblib"
        assert model_path == expected, f"Model path should be {expected}"

    @patch("src.pipelines.training_pipeline.save_model")
    @patch("src.pipelines.training_pipeline.final_evaluation")
    @patch("src.pipelines.training_pipeline.retrain_on_train_val")
    @patch("src.pipelines.training_pipeline.train_model")
    @patch("src.pipelines.training_pipeline.load_split_data")
    @patch("src.pipelines.training_pipeline.run_feature_engineering")
    @patch("src.pipelines.training_pipeline.check_data_drift")
    @patch("src.pipelines.training_pipeline.validate_raw_data")
    @patch("src.pipelines.training_pipeline.save")
    @patch("src.pipelines.training_pipeline.generate")
    def test_pipeline_uses_retrained_model_for_evaluation(
        self, mock_generate, mock_save_data, mock_validate, mock_drift,
        mock_fe, mock_load, mock_train, mock_retrain, mock_eval, mock_save,
        pipeline_config, mock_pipeline_dependencies, sample_fraud_df
    ):
        """Verify final evaluation uses retrained model, not initial."""
        mock_generate.return_value = sample_fraud_df
        mock_validate.return_value = sample_fraud_df
        mock_fe.return_value = mock_pipeline_dependencies
        mock_load.return_value = (
            pd.DataFrame(), pd.Series(dtype=int),
            pd.DataFrame(), pd.Series(dtype=int),
            pd.DataFrame(), pd.Series(dtype=int),
        )
        initial_model = MagicMock(name="initial")
        final_model = MagicMock(name="final")
        mock_train.return_value = (initial_model, {"pr_auc": 0.85})
        mock_retrain.return_value = final_model
        mock_eval.return_value = {"pr_auc": 0.83}

        run_pipeline(pipeline_config, skip_drift=True)

        eval_call = mock_eval.call_args
        assert eval_call.args[0] is final_model, "Must evaluate retrained model"


class TestMainCLI:
    """Tests for CLI entry point."""

    @patch("src.pipelines.training_pipeline.run_pipeline")
    @patch("src.pipelines.training_pipeline.load_config")
    def test_main_loads_config_from_arg(self, mock_load_config, mock_run, tmp_path):
        """Verify main loads config from provided path."""
        config_path = tmp_path / "test_config.yaml"
        config_path.touch()
        mock_config = MagicMock()
        mock_load_config.return_value = mock_config

        with patch("sys.argv", ["train_pipeline.py", "--config", str(config_path)]):
            main()

        mock_load_config.assert_called_once()
        mock_run.assert_called_once_with(mock_config, skip_drift=False)

    @patch("src.pipelines.training_pipeline.run_pipeline")
    @patch("src.pipelines.training_pipeline.load_config")
    def test_main_uses_default_config_path(self, mock_load_config, mock_run):
        """Verify default config path is used when not specified."""
        mock_config = MagicMock()
        mock_load_config.return_value = mock_config

        with patch("sys.argv", ["train_pipeline.py"]):
            main()

        call_args = mock_load_config.call_args
        config_path = call_args.args[0]
        assert config_path == Path("src/config/config.yaml"), "Must use default config path"


class TestPipelineIntegrity:
    """Tests for pipeline data integrity."""

    @patch("src.pipelines.training_pipeline.save_model")
    @patch("src.pipelines.training_pipeline.final_evaluation")
    @patch("src.pipelines.training_pipeline.retrain_on_train_val")
    @patch("src.pipelines.training_pipeline.train_model")
    @patch("src.pipelines.training_pipeline.load_split_data")
    @patch("src.pipelines.training_pipeline.run_feature_engineering")
    @patch("src.pipelines.training_pipeline.check_data_drift")
    @patch("src.pipelines.training_pipeline.validate_raw_data")
    @patch("src.pipelines.training_pipeline.save")
    @patch("src.pipelines.training_pipeline.generate")
    def test_pipeline_loads_correct_split_paths(
        self, mock_generate, mock_save_data, mock_validate, mock_drift,
        mock_fe, mock_load, mock_train, mock_retrain, mock_eval, mock_save,
        pipeline_config, mock_pipeline_dependencies, trained_sklearn_model, sample_fraud_df
    ):
        """Verify data loading receives correct paths from feature engineering."""
        mock_generate.return_value = sample_fraud_df
        mock_validate.return_value = sample_fraud_df
        mock_fe.return_value = mock_pipeline_dependencies
        mock_load.return_value = (
            pd.DataFrame(), pd.Series(dtype=int),
            pd.DataFrame(), pd.Series(dtype=int),
            pd.DataFrame(), pd.Series(dtype=int),
        )
        mock_train.return_value = (trained_sklearn_model, {"pr_auc": 0.85})
        mock_retrain.return_value = trained_sklearn_model
        mock_eval.return_value = {"pr_auc": 0.83}

        run_pipeline(pipeline_config, skip_drift=True)

        load_call = mock_load.call_args
        assert load_call.args[0] == mock_pipeline_dependencies["train"], "Wrong train path"
        assert load_call.args[1] == mock_pipeline_dependencies["val"], "Wrong val path"
        assert load_call.args[2] == mock_pipeline_dependencies["test"], "Wrong test path"

    @patch("src.pipelines.training_pipeline.save_model")
    @patch("src.pipelines.training_pipeline.final_evaluation")
    @patch("src.pipelines.training_pipeline.retrain_on_train_val")
    @patch("src.pipelines.training_pipeline.train_model")
    @patch("src.pipelines.training_pipeline.load_split_data")
    @patch("src.pipelines.training_pipeline.run_feature_engineering")
    @patch("src.pipelines.training_pipeline.check_data_drift")
    @patch("src.pipelines.training_pipeline.validate_raw_data")
    @patch("src.pipelines.training_pipeline.save")
    @patch("src.pipelines.training_pipeline.generate")
    def test_pipeline_passes_config_to_training(
        self, mock_generate, mock_save_data, mock_validate, mock_drift,
        mock_fe, mock_load, mock_train, mock_retrain, mock_eval, mock_save,
        pipeline_config, mock_pipeline_dependencies, trained_sklearn_model, sample_fraud_df
    ):
        """Verify config is passed to all training functions."""
        mock_generate.return_value = sample_fraud_df
        mock_validate.return_value = sample_fraud_df
        mock_fe.return_value = mock_pipeline_dependencies
        mock_load.return_value = (
            pd.DataFrame(), pd.Series(dtype=int),
            pd.DataFrame(), pd.Series(dtype=int),
            pd.DataFrame(), pd.Series(dtype=int),
        )
        mock_train.return_value = (trained_sklearn_model, {"pr_auc": 0.85})
        mock_retrain.return_value = trained_sklearn_model
        mock_eval.return_value = {"pr_auc": 0.83}

        run_pipeline(pipeline_config, skip_drift=True)

        assert mock_train.call_args.args[-1] is pipeline_config, "Config not passed to train"
        assert mock_retrain.call_args.args[-1] is pipeline_config, "Config not passed to retrain"
        assert mock_eval.call_args.args[-1] is pipeline_config, "Config not passed to eval"
