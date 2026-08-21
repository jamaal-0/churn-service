from pathlib import Path

import yaml
from dotenv import load_dotenv

from env_config.environment import get_required_env
from ml.schemas.config import PipelineConfig


load_dotenv()


def load_config(
    path: str | Path,
) -> PipelineConfig:

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    with path.open("r") as file:
        raw_config = yaml.safe_load(file)

    raw_config["mlflow"]["tracking_uri"] = (
        get_required_env("MLFLOW_TRACKING_URI")
    )

    return PipelineConfig.model_validate(
        raw_config
    )