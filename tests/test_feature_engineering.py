import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.feature_engineering import create_features


def test_feature_engineering_shape():
    df = pd.DataFrame({
        "G1": [10, 12],
        "G2": [11, 13],
        "absences": [5, 15],
        "failures": [0, 1],
        "studytime": [2, 3],
        "famrel": [4, 5],
        "freetime": [3, 2],
        "goout": [2, 4]
    })

    result = create_features(df)

    assert len(result) == 2
    assert "assessment_trend" in result.columns
    assert "previous_grade_avg" in result.columns
    assert "absence_band" in result.columns
    assert "failure_risk" in result.columns
    assert "engagement_score" in result.columns


def test_assessment_trend():
    df = pd.DataFrame({
        "G1": [10],
        "G2": [14],
        "absences": [5],
        "failures": [0],
        "studytime": [2],
        "famrel": [4],
        "freetime": [3],
        "goout": [2]
    })

    result = create_features(df)

    assert result["assessment_trend"].iloc[0] == 4


def test_failure_risk():
    df = pd.DataFrame({
        "G1": [10, 10],
        "G2": [12, 12],
        "absences": [5, 5],
        "failures": [0, 2],
        "studytime": [2, 2],
        "famrel": [4, 4],
        "freetime": [3, 3],
        "goout": [2, 2]
    })

    result = create_features(df)

    assert result["failure_risk"].tolist() == [0, 1]