from unittest.mock import patch

from ml.pipeline import TrainingPipeline


@patch("ml.pipeline.log_training_run")
@patch("ml.pipeline.start_run")
def test_training_pipeline(
    mock_start_run,
    mock_log_training_run,
):

    # Make the mocked MLflow context manager work
    mock_start_run.return_value.__enter__.return_value = None

    pipeline = TrainingPipeline()

    metrics = pipeline.run_pipeline(
        "ml/config/teleco.yaml"
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
    mock_start_run.assert_called_once()

    # Training results should have been logged
    mock_log_training_run.assert_called_once()