# ============================================================
# SHAP EXPLAINABILITY - CLASSIFICATION
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

print("===== CLASSIFICATION SHAP =====")


# ============================================================
# 2. CREATE SUPPORT LEVELS
# ============================================================

def support_group(score):

    if score < 10:
        return "High Support"

    elif score < 15:
        return "Medium Support"

    else:
        return "Low Support"


df["Support_Level"] = df["G3"].apply(
    support_group
)


# ============================================================
# 3. PREPARE FEATURES
# ============================================================

X = df.drop(
    columns=["G3", "Support_Level"]
)

y = df["Support_Level"]

print("Dataset shape:", X.shape)

print("\nSupport level distribution:")
print(y.value_counts())


# ============================================================
# 4. LOAD CLASSIFICATION MODEL
# ============================================================

preprocessor = joblib.load(
    "models/classification_preprocessor.pkl"
)

model = joblib.load(
    "models/svm_classifier.pkl"
)

print("\nModel loaded successfully.")
print("Preprocessor loaded successfully.")


# ============================================================
# 5. TRANSFORM DATA
# ============================================================

X_processed = preprocessor.transform(X)

print("\n===== PROCESSED DATA =====")
print("Processed shape:", X_processed.shape)


# Convert sparse matrix to dense
if hasattr(X_processed, "toarray"):
    X_processed = X_processed.toarray()


# Get feature names
feature_names = (
    preprocessor.get_feature_names_out()
)

print(
    "Number of feature names:",
    len(feature_names)
)


# ============================================================
# 6. SELECT BACKGROUND DATA
# ============================================================

background = X_processed[:20]

print("\nBackground samples:", len(background))


# ============================================================
# 7. CREATE SHAP EXPLAINER
# ============================================================

print("\n===== CREATING SHAP EXPLAINER =====")

explainer = shap.KernelExplainer(
    model.predict_proba,
    background
)

print("SHAP KernelExplainer created successfully.")


# ============================================================
# 8. CALCULATE SHAP VALUES
# ============================================================

# Explain only first 10 students
# This keeps Kernel SHAP reasonably fast.

X_explain = X_processed[:3]

print("\nCalculating SHAP values...")
print("Students being explained:", len(X_explain))

shap_values = explainer.shap_values(
    X_explain
)

print("SHAP values generated successfully.")


# ============================================================
# 9. SHOW MODEL CLASSES
# ============================================================

print("\n===== MODEL CLASSES =====")

print(
    "Classes:",
    model.classes_
)


# ============================================================
# 10. INDIVIDUAL STUDENT PREDICTION
# ============================================================

student_index = 0

student_data = X.iloc[
    student_index
]

student_processed = X_processed[
    student_index:student_index + 1
]


prediction = model.predict(
    student_processed
)[0]


probabilities = model.predict_proba(
    student_processed
)[0]


print("\n===== INDIVIDUAL STUDENT =====")

print(
    "Student index:",
    student_index
)

print(
    "\nStudent information:"
)

print(student_data)


print(
    "\nPredicted Support Level:",
    prediction
)


print("\nPrediction probabilities:")

for class_name, probability in zip(
    model.classes_,
    probabilities
):

    print(
        f"{class_name}: {probability:.3f}"
    )


# ============================================================
# 11. FIND HIGH-SUPPORT CLASS
# ============================================================

high_support_index = list(
    model.classes_
).index("High Support")


# ============================================================
# 12. GET SHAP VALUES FOR HIGH SUPPORT
# ============================================================

# SHAP output can have different formats
# depending on the SHAP version.

if isinstance(shap_values, list):

    student_shap_values = (
        shap_values[high_support_index][0]
    )

else:

    if len(shap_values.shape) == 3:

        student_shap_values = (
            shap_values[
                0,
                :,
                high_support_index
            ]
        )

    else:

        student_shap_values = (
            shap_values[0]
        )


# ============================================================
# 13. CREATE FEATURE IMPORTANCE TABLE
# ============================================================

importance = pd.DataFrame({

    "Feature": feature_names,

    "SHAP_Value": student_shap_values

})


importance["Absolute_SHAP"] = (
    importance["SHAP_Value"].abs()
)


importance = importance.sort_values(
    "Absolute_SHAP",
    ascending=False
)


# ============================================================
# 14. DISPLAY TOP FEATURES
# ============================================================

print(
    "\n===== TOP FACTORS FOR HIGH SUPPORT ====="
)

print(
    importance[
        [
            "Feature",
            "SHAP_Value"
        ]
    ].head(10).to_string(
        index=False
    )
)


# ============================================================
# 15. SHAP SUMMARY PLOT
# ============================================================

print(
    "\nCreating SHAP summary plot..."
)

# Use the High Support class
if isinstance(shap_values, list):

    summary_values = shap_values[
        high_support_index
    ]

else:

    if len(shap_values.shape) == 3:

        summary_values = shap_values[
            :,
            :,
            high_support_index
        ]

    else:

        summary_values = shap_values


plt.figure(
    figsize=(10, 6)
)

shap.summary_plot(
    summary_values,
    X_explain,
    feature_names=feature_names,
    show=False
)

plt.title(
    "SHAP Explanation - High Support Classification"
)

plt.tight_layout()

plt.show()


# ============================================================
# END
# ============================================================

print(
    "\n===== CLASSIFICATION SHAP COMPLETED ====="
)