import os
from datetime import datetime, timezone
from typing import Any


class PredictionLogger:
    def __init__(self) -> None:
        self.enabled = os.getenv("ENABLE_ELASTICSEARCH_LOGGING", "false").lower() == "true"
        self.host = os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")
        self.index = os.getenv("ELASTICSEARCH_INDEX", "bloomops-predictions")
        self._client = None

    def _get_client(self):
        if not self.enabled:
            return None
        if self._client is None:
            try:
                from elasticsearch import Elasticsearch

                self._client = Elasticsearch(self.host)
            except Exception:
                self._client = None
        return self._client

    def log_prediction(self, features: dict[str, float], prediction: dict[str, Any]) -> None:
        client = self._get_client()
        if client is None:
            return

        document = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "features": features,
            "prediction": prediction.get("prediction"),
            "label": prediction.get("label"),
            "model_source": prediction.get("model_source"),
        }

        try:
            client.index(index=self.index, document=document)
        except Exception:
            return


prediction_logger = PredictionLogger()
