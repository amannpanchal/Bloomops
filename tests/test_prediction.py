from app.predict import IrisPredictor


def test_fallback_predicts_setosa_when_model_is_missing(tmp_path):
    predictor = IrisPredictor(model_path=str(tmp_path / "missing.joblib"))

    result = predictor.predict([5.1, 3.5, 1.4, 0.2])

    assert result["prediction"] == 0
    assert result["label"] == "setosa"
    assert result["model_source"] == "fallback-rule"


def test_fallback_predicts_virginica_for_large_petals(tmp_path):
    predictor = IrisPredictor(model_path=str(tmp_path / "missing.joblib"))

    result = predictor.predict([6.3, 3.3, 6.0, 2.5])

    assert result["prediction"] == 2
    assert result["label"] == "virginica"

