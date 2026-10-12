# BloomOps API

## Base URL

```text
http://localhost:8000
```

## Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/health` | Returns service health status |
| POST | `/predict` | Returns an Iris class prediction |

## GET /health

### Response Example

```json
{
  "status": "ok"
}
```

## POST /predict

### Request Body

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

### Request Fields

| Field | Type | Description |
| --- | --- | --- |
| `sepal_length` | number | Sepal length in centimeters |
| `sepal_width` | number | Sepal width in centimeters |
| `petal_length` | number | Petal length in centimeters |
| `petal_width` | number | Petal width in centimeters |

### Response Body

```json
{
  "prediction": 0,
  "label": "setosa",
  "model_source": "models/model.joblib"
}
```

### Response Fields

| Field | Type | Description |
| --- | --- | --- |
| `prediction` | integer | Numeric model class |
| `label` | string | Human-readable class label |
| `model_source` | string | Model artifact path or fallback source |

### curl Example

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
