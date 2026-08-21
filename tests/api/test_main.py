from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from env_config.environment import (
    MLFLOW_TRACKING_URI,
    MLFLOW_MODEL_NAME,
    MLFLOW_MODEL_VERSION,
)
from api.main import app


@patch("api.main.ModelService")
def test_health_check(mock_model_service):

    mock_service = MagicMock()
    mock_model_service.return_value = mock_service

    with TestClient(app) as client:

        response = client.post("/")

    assert response.status_code == 200

    assert response.json() == {
        "health": "ok"
    }


@patch("api.main.ModelService")
def test_lifespan_loads_model(
    mock_model_service,
):

    mock_service = MagicMock()
    mock_model_service.return_value = mock_service

    with TestClient(app):

        mock_model_service.assert_called_once_with(
            tracking_uri=MLFLOW_TRACKING_URI,
            model_name=MLFLOW_MODEL_NAME,
            model_version=MLFLOW_MODEL_VERSION,
        )

        mock_service.load_model.assert_called_once()

    assert app.state.model_service is None


@patch("api.main.ModelService")
def test_predict_single(
    mock_model_service,
):

    mock_service = MagicMock()

    mock_X = pd.DataFrame(
        {
            "gender": ["Female"],
            "SeniorCitizen": [0],
            "tenure": [12],
            "MonthlyCharges": [70.5],
        }
    )

    mock_service.validate_input.return_value = mock_X

    mock_service.predict.return_value = np.array([1])

    mock_service.predict_proba.return_value = np.array([
        [0.23, 0.77]
    ])

    mock_model_service.return_value = mock_service

    with TestClient(app) as client:

        response = client.post(
            "/prediction/single",
            json={
                "features": {
                    "gender": "Female",
                    "SeniorCitizen": 0,
                    "tenure": 12,
                    "MonthlyCharges": 70.5,
                }
            },
        )

    assert response.status_code == 200

    assert response.json() == {
        "prediction": 1,
        "probability": 0.77,
    }

    mock_service.validate_input.assert_called_once_with(
        {
            "gender": "Female",
            "SeniorCitizen": 0,
            "tenure": 12,
            "MonthlyCharges": 70.5,
        }
    )

    mock_service.predict.assert_called_once_with(
        mock_X
    )

    mock_service.predict_proba.assert_called_once_with(
        mock_X
    )


@patch("api.main.ModelService")
def test_predict_batch(
    mock_model_service,
):

    mock_service = MagicMock()

    mock_X = pd.DataFrame(
        {
            "gender": [
                "Female",
                "Male",
            ],
            "SeniorCitizen": [
                0,
                1,
            ],
            "tenure": [
                12,
                48,
            ],
            "MonthlyCharges": [
                70.5,
                90.2,
            ],
        }
    )

    mock_service.validate_batch.return_value = mock_X

    mock_service.predict.return_value = np.array([
        1,
        0,
    ])

    mock_service.predict_proba.return_value = np.array([
        [0.23, 0.77],
        [0.81, 0.19],
    ])

    mock_model_service.return_value = mock_service

    with TestClient(app) as client:

        response = client.post(
            "/prediction/batch",
            json={
                "batch": [
                    {
                        "gender": "Female",
                        "SeniorCitizen": 0,
                        "tenure": 12,
                        "MonthlyCharges": 70.5,
                    },
                    {
                        "gender": "Male",
                        "SeniorCitizen": 1,
                        "tenure": 48,
                        "MonthlyCharges": 90.2,
                    },
                ]
            },
        )

    assert response.status_code == 200

    assert response.json() == {
        "predictions": [1, 0],
        "probabilities": [0.77, 0.19],
    }

    mock_service.validate_batch.assert_called_once_with(
        [
            {
                "gender": "Female",
                "SeniorCitizen": 0,
                "tenure": 12,
                "MonthlyCharges": 70.5,
            },
            {
                "gender": "Male",
                "SeniorCitizen": 1,
                "tenure": 48,
                "MonthlyCharges": 90.2,
            },
        ]
    )

    mock_service.predict.assert_called_once_with(
        mock_X
    )

    mock_service.predict_proba.assert_called_once_with(
        mock_X
    )