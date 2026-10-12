import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.joblib"))
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "bloomops-iris-classification")
REGISTERED_MODEL_NAME = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "").strip()


def train_model() -> dict[str, float]:
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42,
        stratify=iris.target,
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "f1_macro": f1_score(y_test, predictions, average="macro"),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="bloomops-random-forest-iris"):
        mlflow.log_params({"model_type": "RandomForestClassifier", "n_estimators": 100})
        mlflow.log_metrics(metrics)
        log_kwargs = {"artifact_path": "model"}
        if REGISTERED_MODEL_NAME:
            log_kwargs["registered_model_name"] = REGISTERED_MODEL_NAME
        mlflow.sklearn.log_model(model, **log_kwargs)

    return metrics


if __name__ == "__main__":
    results = train_model()
    print(f"Model saved to {MODEL_PATH}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1 macro: {results['f1_macro']:.4f}")
    if REGISTERED_MODEL_NAME:
        print(f"Registered model name: {REGISTERED_MODEL_NAME}")
