from typing import List, Any

from pydantic import BaseModel, Field

class CustomerFeatures(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


class SingleInputSchema(BaseModel):
    features: CustomerFeatures

class BatchInputSchema(BaseModel):
    batch : list[CustomerFeatures] = Field(min_length=1,max_length=1000)


class APIConfig(BaseModel):
    model_name: str
    model_version: str
    tracking_uri: str