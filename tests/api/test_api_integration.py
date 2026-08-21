import numpy as np

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

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
def test_single_prediction(
    mock_model_service,
):

    mock_service = MagicMock()

    mock_service.validate_input.return_value = MagicMock()

    mock_service.predict.return_value = np.array([1])

    mock_service.predict_proba.return_value = np.array(
        [[0.25, 0.75]]
    )

    mock_model_service.return_value = mock_service

    with TestClient(app) as client:

        response = client.post(
            "/prediction/single",
            json={
                "features": {
                    "gender": "Female",
                    "SeniorCitizen": 0,
                    "Partner": "Yes",
                    "Dependents": "No",
                    "tenure": 12,
                    "PhoneService": "Yes",
                    "MultipleLines": "No",
                    "InternetService": "DSL",
                    "OnlineSecurity": "No",
                    "OnlineBackup": "Yes",
                    "DeviceProtection": "No",
                    "TechSupport": "No",
                    "StreamingTV": "No",
                    "StreamingMovies": "No",
                    "Contract": "Month-to-month",
                    "PaperlessBilling": "Yes",
                    "PaymentMethod": "Electronic check",
                    "MonthlyCharges": 70.5,
                    "TotalCharges": 846.0,
                }
            },
        )

    assert response.status_code == 200

    assert response.json() == {
        "prediction": 1,
        "probability": 0.75,
    }

    mock_service.validate_input.assert_called_once()

    mock_service.predict.assert_called_once()

    mock_service.predict_proba.assert_called_once()


@patch("api.main.ModelService")
def test_batch_prediction(
    mock_model_service,
):

    mock_service = MagicMock()

    mock_service.validate_batch.return_value = MagicMock()

    mock_service.predict.return_value = np.array(
        [1, 0, 1, 0]
    )

    mock_service.predict_proba.return_value = np.array(
        [
            [0.20, 0.80],
            [0.85, 0.15],
            [0.35, 0.65],
            [0.90, 0.10],
        ]
    )

    mock_model_service.return_value = mock_service

    with TestClient(app) as client:

        response = client.post(
            "/prediction/batch",
            json={
                "batch": [
                    {
                        "gender": "Female",
                        "SeniorCitizen": 0,
                        "Partner": "Yes",
                        "Dependents": "No",
                        "tenure": 12,
                        "PhoneService": "Yes",
                        "MultipleLines": "No",
                        "InternetService": "DSL",
                        "OnlineSecurity": "No",
                        "OnlineBackup": "Yes",
                        "DeviceProtection": "No",
                        "TechSupport": "No",
                        "StreamingTV": "No",
                        "StreamingMovies": "No",
                        "Contract": "Month-to-month",
                        "PaperlessBilling": "Yes",
                        "PaymentMethod": "Electronic check",
                        "MonthlyCharges": 70.5,
                        "TotalCharges": 846.0,
                    },
                    {
                        "gender": "Male",
                        "SeniorCitizen": 0,
                        "Partner": "No",
                        "Dependents": "No",
                        "tenure": 34,
                        "PhoneService": "Yes",
                        "MultipleLines": "No",
                        "InternetService": "DSL",
                        "OnlineSecurity": "Yes",
                        "OnlineBackup": "No",
                        "DeviceProtection": "Yes",
                        "TechSupport": "No",
                        "StreamingTV": "No",
                        "StreamingMovies": "No",
                        "Contract": "One year",
                        "PaperlessBilling": "No",
                        "PaymentMethod": "Mailed check",
                        "MonthlyCharges": 56.95,
                        "TotalCharges": 1889.5,
                    },
                    {
                        "gender": "Male",
                        "SeniorCitizen": 0,
                        "Partner": "No",
                        "Dependents": "No",
                        "tenure": 2,
                        "PhoneService": "Yes",
                        "MultipleLines": "No",
                        "InternetService": "DSL",
                        "OnlineSecurity": "Yes",
                        "OnlineBackup": "Yes",
                        "DeviceProtection": "No",
                        "TechSupport": "No",
                        "StreamingTV": "No",
                        "StreamingMovies": "No",
                        "Contract": "Month-to-month",
                        "PaperlessBilling": "Yes",
                        "PaymentMethod": "Mailed check",
                        "MonthlyCharges": 53.85,
                        "TotalCharges": 108.15,
                    },
                    {
                        "gender": "Female",
                        "SeniorCitizen": 0,
                        "Partner": "No",
                        "Dependents": "No",
                        "tenure": 8,
                        "PhoneService": "Yes",
                        "MultipleLines": "Yes",
                        "InternetService": "Fiber optic",
                        "OnlineSecurity": "No",
                        "OnlineBackup": "No",
                        "DeviceProtection": "Yes",
                        "TechSupport": "No",
                        "StreamingTV": "Yes",
                        "StreamingMovies": "Yes",
                        "Contract": "Month-to-month",
                        "PaperlessBilling": "Yes",
                        "PaymentMethod": "Electronic check",
                        "MonthlyCharges": 99.65,
                        "TotalCharges": 820.5,
                    },
                ]
            },
        )

    assert response.status_code == 200

    assert response.json() == {
        "predictions": [
            1,
            0,
            1,
            0,
        ],
        "probabilities": [
            0.80,
            0.15,
            0.65,
            0.10,
        ],
    }

    mock_service.validate_batch.assert_called_once()

    mock_service.predict.assert_called_once()

    mock_service.predict_proba.assert_called_once()


@patch("api.main.ModelService")
def test_single_prediction_invalid_request(
    mock_model_service,
):

    mock_service = MagicMock()

    mock_service.validate_input.side_effect = ValueError(
        "Invalid type for feature 'SeniorCitizen'."
    )

    mock_model_service.return_value = mock_service

    with TestClient(app) as client:

        response = client.post(
            "/prediction/single",
            json={
                "features": {
                    "gender": "Female",
                    "SeniorCitizen": "invalid",
                    "Partner": "Yes",
                    "Dependents": "No",
                    "tenure": 12,
                    "PhoneService": "Yes",
                    "MultipleLines": "No",
                    "InternetService": "DSL",
                    "OnlineSecurity": "No",
                    "OnlineBackup": "Yes",
                    "DeviceProtection": "No",
                    "TechSupport": "No",
                    "StreamingTV": "No",
                    "StreamingMovies": "No",
                    "Contract": "Month-to-month",
                    "PaperlessBilling": "Yes",
                    "PaymentMethod": "Electronic check",
                    "MonthlyCharges": 70.5,
                    "TotalCharges": 846.0,
                }
            },
        )

    assert response.status_code == 422

    assert response.json() == {
        "detail": "Invalid type for feature 'SeniorCitizen'."
    }