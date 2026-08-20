from unittest.mock import patch

from ml.tracking.mlflow_tracker import (
    start_run,
    log_training_run,
)


@patch("ml.tracking.mlflow_tracker.mlflow.start_run")
@patch("ml.tracking.mlflow_tracker.mlflow.set_experiment")
def test_start_run(
    mock_set_experiment,
    mock_start_run,
):

    start_run(
        experiment_name="churn-service",
        run_name="test-run",
    )

    mock_set_experiment.assert_called_once_with(
        "churn-service"
    )

    mock_start_run.assert_called_once_with(
        run_name="test-run"
    )


@patch("ml.tracking.mlflow_tracker.mlflow.sklearn.log_model")
@patch("ml.tracking.mlflow_tracker.mlflow.log_metrics")
@patch("ml.tracking.mlflow_tracker.mlflow.log_params")
def test_log_training_run(
    mock_log_params,
    mock_log_metrics,
    mock_log_model,
):

    model = object()

    parameters = {
        "model_type": "logistic_regression",
        "max_iter": 1000,
    }

    metrics = {
        "accuracy": 0.8,
        "recall": 0.7,
        "roc_auc": 0.85,
    }

    log_training_run(
        model=model,
        parameters=parameters,
        metrics=metrics,
    )

    mock_log_params.assert_called_once_with(
        parameters
    )

    mock_log_metrics.assert_called_once_with(
        metrics
    )

    mock_log_model.assert_called_once_with(
        model,
        name="model",
    )