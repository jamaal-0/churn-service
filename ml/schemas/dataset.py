from pydantic import BaseModel


class DatasetSchema(BaseModel):
    target: str
    numerical_features: list[str]
    categorical_features: list[str]
    dropped_features: list[str]