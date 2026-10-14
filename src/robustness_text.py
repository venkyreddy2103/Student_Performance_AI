# ============================================================
# ROBUSTNESS TESTING
# ============================================================

import pandas as pd
import joblib
import numpy as np


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(
    "data/student-mat.csv",
    sep=";"
)

X = df.drop(columns=["G3"])


# ============================================================
# 2. LOAD MODEL AND PREPROCESSOR
# ============================================================

preprocessor = joblib.load(
    "models/regression_preprocessor.pkl"
)

model = joblib.load(
    "models/tuned_random_forest_regressor.pkl"
)

print("===== ROBUSTNESS TESTING =====")
print("Model loaded successfully.")


# ============================================================
# 3. NORMAL INPUT
# ============================================================

normal_student = X.iloc[[0]].copy()

normal_processed = preprocessor.transform(
    normal_student
)

normal_prediction = model.predict(
    normal_processed
)[0]

print("\nNormal input prediction:")
print(round(normal_prediction, 2))


# ============================================================
# 4. PERTURBED INPUT
# ============================================================

perturbed_student = normal_student.copy()

# Increase absences by 5
perturbed_student["absences"] = (
    perturbed_student["absences"] + 5
)

perturbed_processed = preprocessor.transform(
    perturbed_student
)

perturbed_prediction = model.predict(
    perturbed_processed
)[0]

print("\nPerturbed input prediction:")
print(round(perturbed_prediction, 2))


# ============================================================
# 5. COMPARE PREDICTIONS
# ============================================================

prediction_change = (
    perturbed_prediction -
    normal_prediction
)

print("\nPrediction change:")
print(round(prediction_change, 2))


# ============================================================
# 6. MISSING INPUT TEST
# ============================================================

missing_student = normal_student.copy()

# Simulate missing numerical input
missing_student["absences"] = np.nan

print("\nTesting missing input...")

try:

    missing_processed = preprocessor.transform(
        missing_student
    )

    missing_prediction = model.predict(
        missing_processed
    )[0]

    print(
        "Missing-input prediction:",
        round(missing_prediction, 2)
    )

    print(
        "Model handled missing input successfully."
    )

except Exception as e:

    print(
        "Missing input was rejected safely."
    )

    print(
        "Reason:",
        str(e)
    )


# ============================================================
# END
# ============================================================

print(
    "\n===== ROBUSTNESS TEST COMPLETED ====="
)