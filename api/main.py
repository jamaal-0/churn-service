from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
import mlflow, mlflow.sklearn
import pandas as pd
from api.services.model_service import ModelService
from api.schemas import SingleInputSchema, BatchInputSchema
from env_config.environment import (
    MLFLOW_TRACKING_URI,
    MLFLOW_MODEL_NAME,
    MLFLOW_MODEL_VERSION
    )

class ModelInfo():
    def __init__(self, model_uri, model, sklearn_model):
        self.model_uri = model_uri
        self.model = model
        self.sklearn_model = sklearn_model

@asynccontextmanager
async def lifespan(app: FastAPI):

    #this is calling the model service call to init the variables 
    #the alternative is to sequentially:
        #init the tracking uri
        #create the model uri from the model name and model version
        #create an object and store these values in them
    # model_service = ModelService(
    #     tracking_uri=MLFLOW_TRACKING_URI,
    #     model_name=MLFLOW_MODEL_NAME,
    #     model_version=MLFLOW_MODEL_VERSION
    #     )
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = (f"models:/{MLFLOW_MODEL_NAME}/{MLFLOW_MODEL_VERSION}")
    
    

    #loads model using the tracking uri and model uri created from the model name and version
    #the altenative is to call mlflow.sklearn.load_model() with the stored model uri
    #and add the model to the object contaning the model_uri
    #model_service.load_model()
    model = mlflow.pyfunc.load_model(model_uri=model_uri)

    # Load sklearn model for probability predictions
    sklearn_model = mlflow.sklearn.load_model(model_uri)

    #stores the model service class with the functions required to perform inference in the app state
    #alternative is to add the object containing the model_uri and the model itself to the app state
    #app.state.model_service = model_service
    model_info = ModelInfo(model_uri=model_uri, model=model, sklearn_model=sklearn_model)
    app.state.model_info = model_info

    yield

    #app.state.model_service = None
    app.state.model_info = None


app = FastAPI(lifespan=lifespan)


#endpoint to carry out a health check on the server 
@app.post("/")
def health_check():
    return {
        "health": "ok"
    }


# @app.post("/prediction/single")
# def predict_single(request: SingleInputSchema):

#     model_service = app.state.model_service

#     try:
#         #this is equivilent to doing signiture validation on the request features 
#         #to ensure the align with the ones contained within the model
#         X = model_service.validate_input(
#             request.features
#         )
#     except ValueError as exc:
#         raise HTTPException(
#             status_code=422,
#             detail=str(exc),
#         )

#     #carries out single inference to obtain both the class and probability
#     prediction = model_service.predict(X)
#     probability = model_service.predict_proba(X)

#     return {
#         "prediction": prediction[0].item(),
#         "probability": probability[0, 1].item(),
#     }

@app.post("/prediction/single")
def predict_single(request: SingleInputSchema):

    model_info = app.state.model_info

    #Convert Pydantic model into dictionary
    features = request.features.model_dump()

    # Convert dictionary into DataFrame
    X = pd.DataFrame([features])
    print(f'x dtypes: {X.dtypes}')
    print(f'x dataframe: {X}')
    print(f'signiture: {model_info.model.metadata.signature}')

    try:
        # PyFunc automatically validates the model signature
        print("Testing PyFunc prediction")
        prediction = model_info.model.predict(X)

        # sklearn interface provides class probabilities
        print("Testing sklearn probability prediction")
        probability = model_info.sklearn_model.predict_proba(X)

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="An error occurred during model inference."
        )

    return {
        "prediction": prediction[0].item(),
        "probability": probability[0, 1].item(),
    }


@app.post("/prediction/batch")
def predict_batch(request: BatchInputSchema):

    model_info = app.state.model_info

    # Convert customer features into a DataFrame
    X = pd.DataFrame([customer.model_dump() for customer in request.batch])
    try:
        # PyFunc automatically validates the batch input
        prediction = model_info.model.predict(X)

        # sklearn interface generates class probabilities
        probability = model_info.sklearn_model.predict_proba(X)

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="An error occurred during batch inference."
        )

    return {
        "predictions": prediction.tolist(),
        "probabilities": probability[:, 1].tolist()
    }

# @app.post("/prediction/batch")
# def predict_batch(request: BatchInputSchema):

#     model_service = app.state.model_service

#     try:
#         #this is equivilent to doing signiture validation on the request features 
#         #to ensure the align with the ones contained within the model
#         X = model_service.validate_batch(
#             request.batch
#         )
#     except ValueError as exc:
#         raise HTTPException(
#             status_code=422,
#             detail=str(exc),
#         )
#     #carries out batch inference to obtain both the classes and probabilities
#     predictions = model_service.predict(X)
#     probabilities = model_service.predict_proba(X)

#     return {
#         "predictions": predictions.tolist(),
#         "probabilities": probabilities[:, 1].tolist(),
#     }
    