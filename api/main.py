from contextlib import asynccontextmanager
from fastapi import FastAPI
from api.services.model_service import ModelService
from api.schemas import SingleInputSchema, BatchInputSchema



@asynccontextmanager
async def lifespan(app: FastAPI):
    model_service = ModelService (
        tracking_uri="http://127.0.0.1:5000",
        model_name="churn-model",
        model_version="2",
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
    X = model_service.validate_input(request.features)
    prediction = model_service.predict(X)

    return {
        "prediction": prediction[0].item()
    }


@app.post("/prediction/batch")
def predict_batch(request: BatchInputSchema):

    model_service = app.state.model_service
    predictions = []

    for features in request.batch:
        X = model_service.validate_input(features)
        prediction = model_service.predict(X)
        predictions.append(prediction[0].item())

    return {
        "predictions": predictions
        }

    