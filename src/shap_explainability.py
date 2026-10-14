# ============================================================
# SHAP EXPLAINABILITY - REGRESSION
# ============================================================

import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv(
    "data/student-mat.csv",
    sep=";"
)

# Target
y = df["G3"]

# Features
X = df.drop(columns=["G3"])

print("===== SHAP EXPLAINABILITY =====")
print("Dataset shape:", X.shape)


# ============================================================
# 2. LOAD SAVED PREPROCESSOR AND MODEL
# ============================================================

preprocessor = joblib.load(
    "models/regression_preprocessor.pkl"
)

model = joblib.load(
    "models/tuned_random_forest_regressor.pkl"
)

print("Model loaded successfully.")
print("Preprocessor loaded successfully.")


# ============================================================
# 3. PREPARE DATA FOR SHAP
# ============================================================

X_processed = preprocessor.transform(X)

print("\n===== SHAP DATA =====")
print("Original shape:", X.shape)
print("Processed shape:", X_processed.shape)

# Get feature names after one-hot encoding
feature_names = preprocessor.get_feature_names_out()

print("Number of feature names:", len(feature_names))


# ============================================================
# 4. CREATE SHAP EXPLAINER
# ============================================================

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_processed)

print("\n===== SHAP EXPLAINER =====")
print("SHAP values generated successfully.")
print("SHAP values shape:", shap_values.shape)


# ============================================================
# 5. GLOBAL SHAP FEATURE IMPORTANCE
# ============================================================

plt.figure(figsize=(10, 6))

shap.summary_plot(
    shap_values,
    X_processed,
    feature_names=feature_names,
    show=False
)

plt.title(
    "SHAP Feature Importance - Random Forest"
)

plt.tight_layout()
plt.show()


# ============================================================
# 6. INDIVIDUAL STUDENT SHAP EXPLANATION
# ============================================================

student_index = 0

student_shap_values = shap_values[student_index]

print("\n===== INDIVIDUAL STUDENT =====")

print("Student index:", student_index)


# ------------------------------------------------------------
# Student's original information
# ------------------------------------------------------------

print("\nStudent information:")

print(
    X.iloc[student_index]
)


# ------------------------------------------------------------
# Model prediction
# ------------------------------------------------------------

student_prediction = model.predict(
    X_processed[student_index:student_index + 1]
)[0]

print(
    "\nPredicted Final Score (G3):",
    round(student_prediction, 2)
)


# ============================================================
# 7. TOP FACTORS INFLUENCING THE PREDICTION
# ============================================================

importance = pd.DataFrame({

    "Feature": feature_names,

    "SHAP_Value": student_shap_values

})


# Absolute SHAP value tells us
# how strongly a feature influenced the prediction

importance["Absolute_SHAP"] = (
    importance["SHAP_Value"].abs()
)


# Sort by importance

importance = importance.sort_values(
    "Absolute_SHAP",
    ascending=False
)


print(
    "\nTop 10 factors influencing this prediction:"
)

print(
    importance[
        ["Feature", "SHAP_Value"]
    ].head(10).to_string(index=False)
)


# ============================================================
# END
# ============================================================

print("\n===== SHAP ANALYSIS COMPLETED =====")
def explain_student(student):
    """
    Generate SHAP explanation for one student.
    """

    X_student = preprocessor.transform(student)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_student)

    feature_names = preprocessor.get_feature_names_out()
    # Make feature names easier to understand
    feature_names = [
        name.replace("remainder__", "")
             .replace("categorical__", "")
        for name in feature_names
    ]

    explanation = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": shap_values[0]
    })

    explanation["Impact"] = explanation["SHAP Value"].apply(
        lambda x: "Increases prediction" if x > 0 else "Decreases prediction"
    )

    explanation["Absolute Impact"] = explanation["SHAP Value"].abs()

    explanation = explanation.sort_values(
        "Absolute Impact",
        ascending=False
    )

    return explanation.head(10)
if __name__ == "__main__":
    student = df.drop(columns=["G3"]).iloc[[0]]

    result = explain_student(student)

    print("\n===== SHAP EXPLANATION =====")
    print(result[["Feature", "SHAP Value", "Impact"]].to_string(index=False))