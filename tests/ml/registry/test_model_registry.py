from unittest.mock import patch

from ml.registry.model_registry import register_model


@patch("ml.registry.model_registry.mlflow.register_model")
def test_register_model(mock_register_model):

    mock_register_model.return_value.version = "1"

    result = register_model(
        run_id="abc123",
        model_name="churn-model",
    )

    mock_register_model.assert_called_once_with(
        model_uri="runs:/abc123/model",
        name="churn-model",
    )

    assert result == "1"