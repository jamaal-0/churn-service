from typing import List, Any

from pydantic import BaseModel

class SingleInputSchema(BaseModel):
    features : dict[str, Any]

class BatchInputSchema(BaseModel):
    batch : list[dict[str, Any]]


class APIConfig(BaseModel):
    model_name: str
    model_version: str
    tracking_uri: str