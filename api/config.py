from .schemas import APIConfig
from ml.config.loader import load_config
import os
from dotenv import load_dotenv


# def load_api_config() -> APIConfig:
#     ml_config = load_config("ml/config/telco.yaml")

#     return APIConfig(
#         model_name="churn-model",
#         model_version="1",
#         tracking_uri=ml_config.mlflow.tracking_uri,
#     )



load_dotenv()


def get_required_env(name: str) -> str:

    value = os.getenv(name)

    if value is None:
        raise RuntimeError(
            f"Required environment variable "
            f"'{name}' is not set."
        )

    return value


MLFLOW_TRACKING_URI = get_required_env(
    "MLFLOW_TRACKING_URI"
)

MLFLOW_MODEL_NAME = get_required_env(
    "MLFLOW_MODEL_NAME"
)

MLFLOW_MODEL_VERSION = get_required_env(
    "MLFLOW_MODEL_VERSION"
)