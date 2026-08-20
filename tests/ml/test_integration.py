from unittest.mock import patch

from ml.pipeline import TrainingPipeline


@patch("ml.pipeline.log_training_run")
@patch("ml.pipeline.infer_signature")
@patch("ml.pipeline.start_run")
def test_training_pipeline(
    mock_start_run,
    mock_infer_signature,
    mock_log_training_run,
):

    # Make the mocked MLflow context manager work
    mock_start_run.return_value.__enter__.return_value = None

    # Mock generated model signature
    mock_signature = object()
    mock_infer_signature.return_value = mock_signature

    pipeline = TrainingPipeline()

    metrics = pipeline.run_pipeline(
        "ml/config/telco.yaml"
    )

    # Pipeline should return evaluation metrics
    assert isinstance(metrics, dict)

    # Expected metrics from our evaluator
    expected_metrics = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    }

    assert expected_metrics.issubset(metrics.keys())

    # All classification metrics should be between 0 and 1
    for value in metrics.values():
        assert 0 <= value <= 1

    # MLflow run should have been started
    mock_start_run.assert_called_once_with(
        tracking_uri="http://127.0.0.1:5000",
        experiment_name="churn-service",
    )

    # Model signature should have been generated
    mock_infer_signature.assert_called_once()

    # Training results should have been logged
    mock_log_training_run.assert_called_once()

    # Verify that the generated signature was passed to MLflow
    log_kwargs = mock_log_training_run.call_args.kwargs

    assert log_kwargs["signature"] is mock_signature