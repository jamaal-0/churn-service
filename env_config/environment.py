import os
from dotenv import load_dotenv




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