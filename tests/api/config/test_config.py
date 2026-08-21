import importlib

import pytest


def test_get_required_env(monkeypatch):

    monkeypatch.setenv(
        "TEST_VARIABLE",
        "test-value",
    )

    import api.config as config

    assert (
        config.get_required_env("TEST_VARIABLE")
        == "test-value"
    )


def test_get_required_env_missing_variable():

    import api.config as config

    with pytest.raises(
        RuntimeError,
        match="Required environment variable 'MISSING_VARIABLE' is not set",
    ):
        config.get_required_env(
            "MISSING_VARIABLE"
        )


def test_environment_configuration(
    monkeypatch,
):

    monkeypatch.setenv(
        "MLFLOW_TRACKING_URI",
        "http://test-mlflow:5000",
    )

    monkeypatch.setenv(
        "MLFLOW_MODEL_NAME",
        "test-model",
    )

    monkeypatch.setenv(
        "MLFLOW_MODEL_VERSION",
        "99",
    )

    import api.config as config

    importlib.reload(config)

    assert (
        config.MLFLOW_TRACKING_URI
        == "http://test-mlflow:5000"
    )

    assert (
        config.MLFLOW_MODEL_NAME
        == "test-model"
    )

    assert (
        config.MLFLOW_MODEL_VERSION
        == "99"
    )