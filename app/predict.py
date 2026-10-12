import os
from functools import lru_cache
from pathlib import Path

import numpy as np


CLASS_LABELS = {
    0: "setosa",
    1: "versicolor",
    2: "virginica",
}


class IrisPredictor:
    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = Path(model_path or os.getenv("MODEL_PATH", "models/model.joblib"))
        self.model = self._load_model()

    def _load_model(self):
        if self.model_path.exists():
            import joblib

            return joblib.load(self.model_path)
        return None

    def predict(self, features: list[float]) -> dict[str, int | str]:
        data = np.array([features], dtype=float)
        if self.model is not None:
            prediction = int(self.model.predict(data)[0])
            source = str(self.model_path)
        else:
            prediction = self._fallback_predict(features)
            source = "fallback-rule"

        return {
            "prediction": prediction,
            "label": CLASS_LABELS.get(prediction, "unknown"),
            "model_source": source,
        }

    @staticmethod
    def _fallback_predict(features: list[float]) -> int:
        petal_length = features[2]
        petal_width = features[3]
        if petal_length < 2.5:
            return 0
        if petal_width < 1.8:
            return 1
        return 2


@lru_cache
def get_predictor() -> IrisPredictor:
    return IrisPredictor()

