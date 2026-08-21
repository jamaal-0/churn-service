from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from api.services.model_service import ModelService
from api.schemas import SingleInputSchema, BatchInputSchema
from env_config.environment import (
    MLFLOW_TRACKING_URI,
    MLFLOW_MODEL_NAME,
    MLFLOW_MODEL_VERSION
    )


@asynccontextmanager
async def lifespan(app: FastAPI):

    model_service = ModelService(
        tracking_uri=MLFLOW_TRACKING_URI,
        model_name=MLFLOW_MODEL_NAME,
        model_version=MLFLOW_MODEL_VERSION
        )

    model_service.load_model()

    app.state.model_service = model_service

    yield

    app.state.model_service = None


app = FastAPI(lifespan=lifespan)


@app.post("/")
def health_check():
    return {
        "health": "ok"
    }


@app.post("/prediction/single")
def predict_single(request: SingleInputSchema):

    model_service = app.state.model_service

    try:
        X = model_service.validate_input(
            request.features
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        )

    prediction = model_service.predict(X)
    probability = model_service.predict_proba(X)

    return {
        "prediction": prediction[0].item(),
        "probability": probability[0, 1].item(),
    }


@app.post("/prediction/batch")
def predict_batch(request: BatchInputSchema):

    model_service = app.state.model_service

    try:
        X = model_service.validate_batch(
            request.batch
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        )

    predictions = model_service.predict(X)
    probabilities = model_service.predict_proba(X)

    return {
        "predictions": predictions.tolist(),
        "probabilities": probabilities[:, 1].tolist(),
    }
    