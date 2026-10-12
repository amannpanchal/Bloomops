# BloomOps Architecture

## Dataset Flow

The current project uses the built-in Iris dataset from scikit-learn. This keeps the repository clean because no private dataset or large raw data file needs to be committed.

Flow:

```text
scikit-learn dataset loader
-> train/test split
-> model training
-> metric calculation
-> MLflow logging
-> local model artifact
```

In a production version, the `data/` directory can hold small sample datasets or schema examples, while large datasets should be stored in object storage or a data platform.

## Training Pipeline

The training pipeline is implemented in `training/train.py`.

Main steps:

1. Load the dataset.
2. Split the data into training and test sets.
3. Train a `RandomForestClassifier`.
4. Evaluate the model with accuracy and macro F1 score.
5. Save the trained model to `models/model.joblib`.
6. Log parameters, metrics, and model artifacts to MLflow.

The pipeline is intentionally simple so the MLOps structure is easy to understand and extend.

## MLflow Experiment Tracking

MLflow is used to track:

- Experiment name
- Run name
- Model type
- Hyperparameters
- Evaluation metrics
- Model artifacts

The default experiment name is:

```text
bloomops-iris-classification
```

It can be changed with the `MLFLOW_EXPERIMENT_NAME` environment variable.

## Model Registry and Versioning

The project logs trained models through `mlflow.sklearn.log_model`. In a production workflow, the next step would be to register the best model in the MLflow Model Registry.

Recommended versioning flow:

```text
New training run
-> compare metrics
-> register candidate model
-> promote to staging
-> validate with tests
-> promote to production
-> deploy API with selected model version
```

Local model binaries are ignored by Git because they can be regenerated from the training pipeline or stored in MLflow artifact storage.

## FastAPI Inference Service

The inference service is implemented in `app/main.py` and `app/predict.py`.

Available service flow:

```text
Client request
-> FastAPI validation with Pydantic
-> predictor loads model artifact
-> model returns class id
-> API returns class id, label, and model source
```

If `models/model.joblib` is missing, the API uses a deterministic fallback rule. This keeps the API and tests runnable before the first training run.

## Docker Deployment

The project includes:

- `Dockerfile` for the FastAPI service
- `docker-compose.yml` for FastAPI and MLflow
- `.dockerignore` to avoid copying local caches, logs, environments, and model artifacts into images

Docker Compose starts:

- FastAPI on port `8000`
- MLflow UI on port `5000`

The `models/`, `mlruns/`, and `mlartifacts/` directories are local runtime outputs and should not be committed.
