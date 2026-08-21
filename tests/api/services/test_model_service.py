from unittest.mock import MagicMock, patch

import pytest

from api.services.model_service import ModelService


@patch(
    "api.services.model_service.SignatureValidator"
)
@patch(
    "api.services.model_service.get_model_info"
)
@patch(
    "api.services.model_service.mlflow.sklearn.load_model"
)
@patch(
    "api.services.model_service.mlflow.set_tracking_uri"
)
def test_load_model(
    mock_set_tracking_uri,
    mock_load_model,
    mock_get_model_info,
    mock_signature_validator,
):

    mock_model = MagicMock()
    mock_signature = MagicMock()
    mock_model_info = MagicMock()
    mock_validator = MagicMock()

    mock_load_model.return_value = mock_model

    mock_model_info.signature = mock_signature
    mock_get_model_info.return_value = mock_model_info

    mock_signature_validator.return_value = mock_validator

    service = ModelService(
        tracking_uri="http://127.0.0.1:5000",
        model_name="churn-model",
        model_version="1",
    )

    service.load_model()

    mock_set_tracking_uri.assert_called_once_with(
        "http://127.0.0.1:5000"
    )

    mock_load_model.assert_called_once_with(
        "models:/churn-model/1"
    )

    mock_get_model_info.assert_called_once_with(
        "models:/churn-model/1"
    )

    mock_signature_validator.assert_called_once_with(
        mock_signature
    )

    assert service.model is mock_model
    assert service.signature is mock_signature
    assert service.validator is mock_validator


def test_validate_input_without_model():

    service = ModelService(
        tracking_uri="http://127.0.0.1:5000",
        model_name="churn-model",
        model_version="1",
    )

    with pytest.raises(
        RuntimeError,
        match="Model has not been loaded",
    ):
        service.validate_input({})


def test_validate_batch_without_model():

    service = ModelService(
        tracking_uri="http://127.0.0.1:5000",
        model_name="churn-model",
        model_version="1",
    )

    with pytest.raises(
        RuntimeError,
        match="Model has not been loaded",
    ):
        service.validate_batch([])


def test_predict_without_model():

    service = ModelService(
        tracking_uri="http://127.0.0.1:5000",
        model_name="churn-model",
        model_version="1",
    )

    with pytest.raises(
        RuntimeError,
        match="Model has not been loaded",
    ):
        service.predict(None)


def test_predict_proba_without_model():

    service = ModelService(
        tracking_uri="http://127.0.0.1:5000",
        model_name="churn-model",
        model_version="1",
    )

    with pytest.raises(
        RuntimeError,
        match="Model has not been loaded",
    ):
        service.predict_proba(None)


def test_validate_input():

    service = ModelService(
        tracking_uri="http://127.0.0.1:5000",
        model_name="churn-model",
        model_version="1",
    )

    service.validator = MagicMock()

    features = {
        "gender": "Female",
        "tenure": 12,
    }

    expected_result = MagicMock()

    service.validator.validate.return_value = (
        expected_result
    )

    result = service.validate_input(features)

    service.validator.validate.assert_called_once_with(
        features
    )

    assert result is expected_result


def test_validate_batch():

    service = ModelService(
        tracking_uri="http://127.0.0.1:5000",
        model_name="churn-model",
        model_version="1",
    )

    service.validator = MagicMock()

    batch = [
        {
            "gender": "Female",
            "tenure": 12,
        },
        {
            "gender": "Male",
            "tenure": 48,
        },
    ]

    expected_result = MagicMock()

    service.validator.validate_batch.return_value = (
        expected_result
    )

    result = service.validate_batch(batch)

    service.validator.validate_batch.assert_called_once_with(
        batch
    )

    assert result is expected_result