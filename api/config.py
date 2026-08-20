from schemas import APIConfig
from ml.config.loader import load_config


def load_api_config() -> APIConfig:
    ml_config = load_config("ml/config/telco.yaml")

    return APIConfig(
        model_name="churn-model",
        model_version="1",
        tracking_uri=ml_config.mlflow.tracking_uri,
    )