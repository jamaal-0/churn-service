import pytest
import yaml
from pydantic import ValidationError

from ml.config.loader import load_config


def test_load_config_valid(tmp_path):

    config_data = {
        "data": {
            "path": "data/telco_churn.csv",
            "target": "Churn",
            "drop_columns": ["customerID"],
        },
        "problem": {
            "type": "classification",
        },
        "split": {
            "test_size": 0.2,
            "random_state": 42,
        },
        "model": {
            "type": "logistic_regression",
            "parameters": {
                "max_iter": 1000,
            },
        },
        "mlflow": {
            "experiment_name": "churn-service",
        },
    }

    config_path = tmp_path / "config.yaml"

    with config_path.open("w") as file:
        yaml.safe_dump(config_data, file)

    config = load_config(config_path)

    assert config.data.target == "Churn"
    assert config.problem.type == "classification"
    assert config.split.test_size == 0.2
    assert config.model.type == "logistic_regression"
    assert config.mlflow.experiment_name == "churn-service"


def test_load_config_missing_file(tmp_path):

    config_path = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError):
        load_config(config_path)


def test_load_config_invalid_schema(tmp_path):

    config_data = {
        "data": {
            "path": "data/telco_churn.csv",
            "target": "Churn",
            "drop_columns": ["customerID"],
        },
        "problem": {
            "type": "classification",
        },
        "split": {
            "test_size": 2,
            "random_state": 42,
        },
        "model": {
            "type": "logistic_regression",
            "parameters": {},
        },
        "mlflow": {
            "experiment_name": "churn-service",
        },
    }

    config_path = tmp_path / "invalid.yaml"

    with config_path.open("w") as file:
        yaml.safe_dump(config_data, file)

    with pytest.raises(ValidationError):
        load_config(config_path)