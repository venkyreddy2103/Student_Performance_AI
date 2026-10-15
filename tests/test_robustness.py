import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd

from src.feature_engineering import create_features


DATA_PATH = PROJECT_ROOT / "data" / "student-mat.csv"
PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "regression_preprocessor.pkl"
MODEL_PATH = PROJECT_ROOT / "models" / "tuned_random_forest_regressor.pkl"


def load_model_and_data():
    df = pd.read_csv(DATA_PATH, sep=";")

    X = df.drop(columns=["G3"])
    X = create_features(X)

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)

    return X, preprocessor, model


def predict_student(X, preprocessor, model, index=0):
    student = X.iloc[[index]]
    processed = preprocessor.transform(student)
    return model.predict(processed)[0]


def test_prediction_with_missing_absences():
    X, preprocessor, model = load_model_and_data()

    X_modified = X.copy()
    X_modified.loc[X_modified.index[0], "absences"] = 0

    prediction = predict_student(
        X_modified, preprocessor, model, 0
    )

    assert pd.notna(prediction)
    assert 0 <= prediction <= 20


def test_prediction_after_increasing_g2():
    X, preprocessor, model = load_model_and_data()

    original_prediction = predict_student(
        X, preprocessor, model, 0
    )

    X_modified = X.copy()
    X_modified.loc[X_modified.index[0], "G2"] += 1

    modified_prediction = predict_student(
        X_modified, preprocessor, model, 0
    )

    assert pd.notna(modified_prediction)
    assert 0 <= modified_prediction <= 20
    assert abs(modified_prediction - original_prediction) <= 20


def test_prediction_after_increasing_g1():
    X, preprocessor, model = load_model_and_data()

    original_prediction = predict_student(
        X, preprocessor, model, 0
    )

    X_modified = X.copy()
    X_modified.loc[X_modified.index[0], "G1"] += 1

    modified_prediction = predict_student(
        X_modified, preprocessor, model, 0
    )

    assert pd.notna(modified_prediction)
    assert 0 <= modified_prediction <= 20
    assert abs(modified_prediction - original_prediction) <= 20