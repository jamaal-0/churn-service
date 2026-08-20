from pathlib import Path
import yaml
from ml.schemas.config import PipelineConfig


def load_config(path: str | Path) -> PipelineConfig:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
            )

    with path.open("r") as file:
        raw_config = yaml.safe_load(file)

    return PipelineConfig.model_validate(raw_config)