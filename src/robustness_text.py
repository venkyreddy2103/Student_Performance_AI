# ============================================================
# ROBUSTNESS TESTING
# Student Performance AI Project
# ============================================================

import os
import pandas as pd
import joblib

from feature_engineering import create_features


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "student-mat.csv"
)

PREPROCESSOR_PATH = os.path.join(
    BASE_DIR,
    "models",
    "regression_preprocessor.pkl"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "tuned_random_forest_regressor.pkl"
)

REPORTS_DIR = os.path.join(
    BASE_DIR,
    "reports"
)

os.makedirs(
    REPORTS_DIR,
    exist_ok=True
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 50)
print("        ROBUSTNESS TESTING")
print("=" * 50)

df = pd.read_csv(
    DATA_PATH,
    sep=";"
)

print()
print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)


# ============================================================
# 3. LOAD MODEL
# ============================================================

preprocessor = joblib.load(
    PREPROCESSOR_PATH
)

model = joblib.load(
    MODEL_PATH
)

print()
print("Model loaded successfully.")


# ============================================================
# 4. SELECT STUDENT
# ============================================================

student_index = 0

student = (
    df.drop(columns=["G3"])
    .iloc[[student_index]]
    .copy()
)


# ============================================================
# 5. PREDICTION FUNCTION
# ============================================================

def predict_student(student_data):

    engineered_data = create_features(
        student_data.copy()
    )

    processed_data = (
        preprocessor.transform(
            engineered_data
        )
    )

    if hasattr(processed_data, "toarray"):
        processed_data = processed_data.toarray()

    prediction = model.predict(
        processed_data
    )[0]

    return float(prediction)


# ============================================================
# 6. NORMAL PREDICTION
# ============================================================

print()
print("===== TEST 1: NORMAL INPUT =====")

normal_prediction = predict_student(
    student
)

print(
    "Normal prediction:",
    round(normal_prediction, 2)
)


# ============================================================
# 7. PERTURBED ABSENCES
# ============================================================

print()
print("===== TEST 2: ABSENCES +5 =====")

perturbed_student = student.copy()

original_absences = (
    perturbed_student["absences"].iloc[0]
)

perturbed_student.loc[
    perturbed_student.index[0],
    "absences"
] = original_absences + 5


perturbed_prediction = predict_student(
    perturbed_student
)

change = (
    perturbed_prediction -
    normal_prediction
)

print(
    "Original absences:",
    original_absences
)

print(
    "Modified absences:",
    original_absences + 5
)

print(
    "Modified prediction:",
    round(
        perturbed_prediction,
        2
    )
)

print(
    "Prediction change:",
    round(
        change,
        2
    )
)


# ============================================================
# 8. MISSING INPUT
# ============================================================

print()
print("===== TEST 3: MISSING ABSENCES =====")

missing_student = student.copy()

missing_student.loc[
    missing_student.index[0],
    "absences"
] = None


try:

    missing_prediction = predict_student(
        missing_student
    )

    print(
        "Missing-input prediction:",
        round(
            missing_prediction,
            2
        )
    )

    missing_status = "PASSED"

except Exception as error:

    print(
        "Missing-input test failed:"
    )

    print(error)

    missing_status = "FAILED"


# ============================================================
# 9. G1 PERTURBATION
# ============================================================

print()
print("===== TEST 4: G1 +1 =====")

g1_student = student.copy()

original_g1 = (
    g1_student["G1"].iloc[0]
)

g1_student.loc[
    g1_student.index[0],
    "G1"
] = min(
    original_g1 + 1,
    20
)

g1_prediction = predict_student(
    g1_student
)

g1_change = (
    g1_prediction -
    normal_prediction
)

print(
    "Original G1:",
    original_g1
)

print(
    "Modified G1:",
    min(original_g1 + 1, 20)
)

print(
    "Modified prediction:",
    round(
        g1_prediction,
        2
    )
)

print(
    "Prediction change:",
    round(
        g1_change,
        2
    )
)


# ============================================================
# 10. G2 PERTURBATION
# ============================================================

print()
print("===== TEST 5: G2 +1 =====")

g2_student = student.copy()

original_g2 = (
    g2_student["G2"].iloc[0]
)

g2_student.loc[
    g2_student.index[0],
    "G2"
] = min(
    original_g2 + 1,
    20
)

g2_prediction = predict_student(
    g2_student
)

g2_change = (
    g2_prediction -
    normal_prediction
)

print(
    "Original G2:",
    original_g2
)

print(
    "Modified G2:",
    min(original_g2 + 1, 20)
)

print(
    "Modified prediction:",
    round(
        g2_prediction,
        2
    )
)

print(
    "Prediction change:",
    round(
        g2_change,
        2
    )
)


# ============================================================
# 11. CREATE RESULTS TABLE
# ============================================================

results = pd.DataFrame({

    "Test": [
        "Normal input",
        "Absences +5",
        "Missing absences",
        "G1 +1",
        "G2 +1"
    ],

    "Prediction": [
        normal_prediction,
        perturbed_prediction,
        missing_prediction
        if missing_status == "PASSED"
        else None,
        g1_prediction,
        g2_prediction
    ]
})


# ============================================================
# 12. SAVE RESULTS
# ============================================================

output_path = os.path.join(
    REPORTS_DIR,
    "robustness_test_results.csv"
)

results.to_csv(
    output_path,
    index=False
)


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print()
print("=" * 50)
print("ROBUSTNESS TEST SUMMARY")
print("=" * 50)

print(
    "Normal prediction:",
    round(
        normal_prediction,
        2
    )
)

print(
    "Absences +5 change:",
    round(
        change,
        2
    )
)

print(
    "Missing input test:",
    missing_status
)

print(
    "G1 +1 change:",
    round(
        g1_change,
        2
    )
)

print(
    "G2 +1 change:",
    round(
        g2_change,
        2
    )
)

print()
print(
    "Results saved to:"
)

print(
    output_path
)

print()
print("=" * 50)
print("ROBUSTNESS TESTING COMPLETED")
print("=" * 50)