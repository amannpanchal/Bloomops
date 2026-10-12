from fastapi import FastAPI

from app.monitoring import prediction_logger
from app.predict import get_predictor
from app.schemas import PredictionRequest, PredictionResponse


app = FastAPI(
    title="BloomOps",
    description="Grow machine learning from experiment to production with model training, MLflow tracking, and Iris classification inference.",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    feature_values = {
        "sepal_length": payload.sepal_length,
        "sepal_width": payload.sepal_width,
        "petal_length": payload.petal_length,
        "petal_width": payload.petal_width,
    }
    result = get_predictor().predict(list(feature_values.values()))
    prediction_logger.log_prediction(feature_values, result)
    return PredictionResponse(**result)
