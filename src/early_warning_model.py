# ============================================================
# EARLY WARNING STUDENT PERFORMANCE MODEL
# ============================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(
    "data/student-mat.csv",
    sep=";"
)

print("===== EARLY WARNING MODEL =====")
print("Original dataset shape:", df.shape)


# ============================================================
# 2. REMOVE G1 AND G2
# ============================================================

# G1 and G2 are previous-period grades.
# We remove them to simulate an earlier intervention scenario.

X = df.drop(
    columns=["G3", "G1", "G2"]
)

y = df["G3"]

print("\nFeatures used:", X.shape[1])
print("G1 and G2 removed.")


# ============================================================
# 3. IDENTIFY FEATURE TYPES
# ============================================================

categorical_features = X.select_dtypes(
    include=["object"]
).columns

numerical_features = X.select_dtypes(
    exclude=["object"]
).columns


# ============================================================
# 4. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )

    ],

    remainder="passthrough"
)


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42
)


# ============================================================
# 6. TRANSFORM DATA
# ============================================================

X_train_processed = preprocessor.fit_transform(
    X_train
)

X_test_processed = preprocessor.transform(
    X_test
)


# ============================================================
# 7. TRAIN RANDOM FOREST
# ============================================================

model = RandomForestRegressor(

    n_estimators=300,

    max_depth=10,

    random_state=42
)


model.fit(
    X_train_processed,
    y_train
)


# ============================================================
# 8. PREDICTION
# ============================================================

predictions = model.predict(
    X_test_processed
)


# ============================================================
# 9. EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)


print("\n===== EARLY WARNING RESULTS =====")

print(
    "MAE:",
    round(mae, 4)
)

print(
    "RMSE:",
    round(rmse, 4)
)

print(
    "R²:",
    round(r2, 4)
)


# ============================================================
# 10. COMPARE WITH FULL MODEL
# ============================================================

print("\n===== MODEL COMPARISON =====")

print("Full model uses G1 + G2.")
print("Early-warning model excludes G1 + G2.")

print(
    "\nEarly-warning model evaluation completed."
)


# ============================================================
# END
# ============================================================

print(
    "\n===== EARLY WARNING MODEL COMPLETED ====="
)