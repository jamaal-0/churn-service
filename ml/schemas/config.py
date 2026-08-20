from typing import Any, Literal
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    path: str
    target: str
    drop_columns: list[str] = []


class ProblemConfig(BaseModel):
    type: Literal["classification", "regression"]


class SplitConfig(BaseModel):
    test_size: float = Field(gt=0, lt=1)
    random_state: int


class ModelConfig(BaseModel):
    type: str
    parameters: dict[str, Any] = {}


class MLflowConfig(BaseModel):
    experiment_name: str


class PipelineConfig(BaseModel):
    data: DataConfig
    problem: ProblemConfig
    split: SplitConfig
    model: ModelConfig
    mlflow: MLflowConfig