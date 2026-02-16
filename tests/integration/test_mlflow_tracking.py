# tests/integration/test_mlflow_tracking.py
"""Integration tests for MLflow experiment tracking."""

import mlflow
import pytest


@pytest.mark.integration
class TestMLflowExperimentTracking:
    """Tests for MLflow experiment and run management."""

    def test_experiment_is_created(self, mlflow_tracking_uri):
        """Verify experiment is created with correct name."""
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        experiment_name = "test_fraud_detection"

        mlflow.set_experiment(experiment_name)
        experiment = mlflow.get_experiment_by_name(experiment_name)

        assert experiment is not None, "Experiment must be created"
        assert experiment.name == experiment_name, "Experiment name must match"

    def test_run_logs_parameters(self, mlflow_tracking_uri):
        """Verify parameters are logged correctly to run."""
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment("test_params")

        params = {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1}
        with mlflow.start_run() as run:
            mlflow.log_params(params)
            run_id = run.info.run_id

        logged_run = mlflow.get_run(run_id)
        for key, value in params.items():
            assert key in logged_run.data.params, f"Param {key} must be logged"
            assert logged_run.data.params[key] == str(value), f"Param {key} value mismatch"

    def test_run_logs_metrics(self, mlflow_tracking_uri):
        """Verify metrics are logged correctly to run."""
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment("test_metrics")

        metrics = {"precision": 0.85, "recall": 0.72, "f1_score": 0.78, "pr_auc": 0.81}
        with mlflow.start_run() as run:
            mlflow.log_metrics(metrics)
            run_id = run.info.run_id

        logged_run = mlflow.get_run(run_id)
        for key, value in metrics.items():
            assert key in logged_run.data.metrics, f"Metric {key} must be logged"
            assert abs(logged_run.data.metrics[key] - value) < 1e-6, f"Metric {key} mismatch"


@pytest.mark.integration
class TestMLflowModelLogging:
    """Tests for MLflow model artifact logging."""

    def test_model_is_logged_as_artifact(
        self, mlflow_tracking_uri, trained_sklearn_model, numeric_feature_df
    ):
        """Verify model is logged and retrievable."""
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment("test_model_logging")

        with mlflow.start_run() as run:
            mlflow.sklearn.log_model(trained_sklearn_model, "model")
            run_id = run.info.run_id

        model_uri = f"runs:/{run_id}/model"
        loaded = mlflow.sklearn.load_model(model_uri)
        preds = loaded.predict(numeric_feature_df)

        assert len(preds) == len(numeric_feature_df), "Loaded model must predict"

    def test_model_signature_is_logged(
        self, mlflow_tracking_uri, trained_sklearn_model, numeric_feature_df
    ):
        """Verify model signature captures input/output schema."""
        from mlflow.models import infer_signature
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment("test_signature")

        signature = infer_signature(
            numeric_feature_df, trained_sklearn_model.predict(numeric_feature_df)
        )
        with mlflow.start_run() as run:
            mlflow.sklearn.log_model(trained_sklearn_model, "model", signature=signature)
            run_id = run.info.run_id

        model_uri = f"runs:/{run_id}/model"
        model_info = mlflow.models.get_model_info(model_uri)

        assert model_info.signature is not None, "Signature must be logged"


@pytest.mark.integration
class TestMLflowArtifacts:
    """Tests for MLflow artifact management."""

    def test_figure_artifact_is_saved(self, mlflow_tracking_uri, tmp_path):
        """Verify figure artifacts are saved correctly."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment("test_artifacts")

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])

        with mlflow.start_run() as run:
            mlflow.log_figure(fig, "test_plot.png")
            run_id = run.info.run_id
        plt.close(fig)

        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run_id)
        artifact_names = [a.path for a in artifacts]

        assert "test_plot.png" in artifact_names, "Figure must be logged as artifact"

    def test_multiple_metrics_across_steps(self, mlflow_tracking_uri):
        """Verify metrics from different pipeline stages are all captured."""
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment("test_multi_stage")

        with mlflow.start_run() as run:
            mlflow.log_metrics({"val_precision": 0.82, "val_recall": 0.75})
            mlflow.log_metrics({"test_precision": 0.80, "test_recall": 0.73})
            run_id = run.info.run_id

        logged_run = mlflow.get_run(run_id)
        metrics = logged_run.data.metrics

        assert "val_precision" in metrics, "Validation metrics must be logged"
        assert "test_precision" in metrics, "Test metrics must be logged"
        assert len(metrics) == 4, "All metrics must be captured"
