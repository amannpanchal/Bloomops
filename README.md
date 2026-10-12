# BloomOps

### Grow machine learning from experiment to production.

BloomOps is a compact, end-to-end MLOps platform for training, tracking, and serving machine learning models. It demonstrates a reproducible workflow using an Iris classifier, MLflow experiment tracking, a FastAPI inference service, and Docker Compose.

> **Project identity:** BloomOps  
> **Suggested GitHub repository name:** `bloomops`

## What it does

- Trains a Random Forest classifier on the scikit-learn Iris dataset.
- Tracks model parameters, metrics, and artifacts with MLflow.
- Exposes health and prediction endpoints through FastAPI.
- Optionally logs predictions to Elasticsearch.
- Runs locally or as a Docker Compose stack.

The sample dataset keeps the project easy to run without private data or large binaries.

## How it fits together

```text
Iris dataset -> Training -> MLflow tracking -> Model artifact -> BloomOps API -> Prediction
```

## Technology

- Python 3.11
- FastAPI and Uvicorn
- scikit-learn, pandas, and NumPy
- MLflow
- pytest
- Docker and Docker Compose
- Optional Elasticsearch and Kibana prediction monitoring

## Project layout

```text
.
├── app/
│   ├── main.py
│   ├── monitoring.py
│   ├── predict.py
│   └── schemas.py
├── data/
├── docs/
│   ├── api.md
│   └── architecture.md
├── models/
├── screenshots/
├── tests/
├── training/
│   └── train.py
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── requirements.txt
```

## Getting started

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

## Train and run

Train the model and start the API:

```bash
make train
make api
```

The API is available at `http://localhost:8000`. Explore its interactive documentation at `http://localhost:8000/docs` and check service health at `http://localhost:8000/health`.

To run the full stack instead:

```bash
make docker-up
```

Docker Compose starts:

- BloomOps API: `http://localhost:8000`
- MLflow UI: `http://localhost:5000`
- Elasticsearch: `http://localhost:9200`
- Kibana: `http://localhost:5601`

Stop the stack with `make docker-down`.

## Make a prediction

Send a request to `POST http://localhost:8000/predict`:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```

Example response:

```json
{
  "prediction": 0,
  "label": "setosa",
  "model_source": "models/model.joblib"
}
```

If a trained model is not present, the API uses a deterministic fallback rule so inference remains testable.

## Configuration

Copy `.env.example` to `.env` to customize model paths, MLflow tracking, and optional Elasticsearch logging. Elasticsearch logging is disabled by default for local runs.

## Tests

```bash
make test
```

## Documentation

- [API reference](docs/api.md)
- [System architecture](docs/architecture.md)

## Next steps

- Add continuous integration and deployment with GitHub Actions.
- Validate models before registration and deployment.
- Add batch inference and model drift monitoring.
- Add authentication and rate limiting before production exposure.

