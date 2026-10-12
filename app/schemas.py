from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    sepal_length: float = Field(..., json_schema_extra={"example": 5.1})
    sepal_width: float = Field(..., json_schema_extra={"example": 3.5})
    petal_length: float = Field(..., json_schema_extra={"example": 1.4})
    petal_width: float = Field(..., json_schema_extra={"example": 0.2})


class PredictionResponse(BaseModel):
    prediction: int = Field(..., json_schema_extra={"example": 0})
    label: str = Field(..., json_schema_extra={"example": "setosa"})
    model_source: str = Field(..., json_schema_extra={"example": "models/model.joblib"})