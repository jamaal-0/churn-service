import importlib

import pytest


def test_get_required_env(monkeypatch):

    monkeypatch.setenv(
        "TEST_VARIABLE",
        "test-value",
    )

    import env_config.environment as environment

    assert (
        environment.get_required_env("TEST_VARIABLE")
        == "test-value"
    )


def test_get_required_env_missing_variable():

    import env_config.environment as environment

    with pytest.raises(
        RuntimeError,
        match="Required environment variable 'MISSING_VARIABLE' is not set",
    ):
        environment.get_required_env(
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

    import env_config.environment as environment

    importlib.reload(environment)

    assert (
        environment.MLFLOW_TRACKING_URI
        == "http://test-mlflow:5000"
    )

    assert (
        environment.MLFLOW_MODEL_NAME
        == "test-model"
    )

    assert (
        environment.MLFLOW_MODEL_VERSION
        == "99"
    )