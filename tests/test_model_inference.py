import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd

from src.feature_engineering import create_features


def test_regression_model_inference():
    data_path = PROJECT_ROOT / "data" / "student-mat.csv"
    preprocessor_path = PROJECT_ROOT / "models" / "regression_preprocessor.pkl"
    model_path = PROJECT_ROOT / "models" / "tuned_random_forest_regressor.pkl"

    df = pd.read_csv(data_path, sep=";")

    X = df.drop(columns=["G3"])
    X = create_features(X)

    preprocessor = joblib.load(preprocessor_path)
    model = joblib.load(model_path)

    X_processed = preprocessor.transform(X)
    predictions = model.predict(X_processed)

    assert len(predictions) == len(X)
    assert all(pd.notna(predictions))
    assert all(predictions >= 0)
    assert all(predictions <= 20)


def test_single_student_prediction():
    data_path = PROJECT_ROOT / "data" / "student-mat.csv"
    preprocessor_path = PROJECT_ROOT / "models" / "regression_preprocessor.pkl"
    model_path = PROJECT_ROOT / "models" / "tuned_random_forest_regressor.pkl"

    df = pd.read_csv(data_path, sep=";")

    X = df.drop(columns=["G3"])
    X = create_features(X)

    preprocessor = joblib.load(preprocessor_path)
    model = joblib.load(model_path)

    X_processed = preprocessor.transform(X.iloc[[0]])
    prediction = model.predict(X_processed)[0]

    assert pd.notna(prediction)
    assert 0 <= prediction <= 20