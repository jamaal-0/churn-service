import mlflow
import mlflow.sklearn
from mlflow.models import get_model_info
from api.services.signature_validator import SignatureValidator


class ModelService:

    def __init__(
        self,
        tracking_uri: str,
        model_name: str,
        model_version: str,
    ):
        self.tracking_uri = tracking_uri
        self.model_name = model_name
        self.model_version = model_version

        self.model = None
        self.signature = None
        self.validator = None

    def load_model(self) -> None:

        mlflow.set_tracking_uri(
            self.tracking_uri
        )

        model_uri = (
            f"models:/{self.model_name}/{self.model_version}"
        )

        self.model = mlflow.sklearn.load_model(
            model_uri
        )

        model_info = get_model_info(
            model_uri
        )

        self.signature = model_info.signature

        self.validator = SignatureValidator(
            self.signature
        )

    def validate_input(
        self,
        features: dict,
    ):
        if self.validator is None:
            raise RuntimeError(
                "Model has not been loaded."
            )

        return self.validator.validate(
            features
        )

    def predict(self, X):

        if self.model is None:
            raise RuntimeError(
                "Model has not been loaded."
            )

        return self.model.predict(X)

    def predict_proba(self, X):

        if self.model is None:
            raise RuntimeError(
                "Model has not been loaded."
            )

        return self.model.predict_proba(X)

    def validate_batch(
        self,
        batch: list[dict],
    ):
        if self.validator is None:
            raise RuntimeError(
                "Model has not been loaded."
            )

        return self.validator.validate_batch(
            batch
        )