# =========================================================
# SHAP EXPLAINABILITY
# Intelligent Student Performance AI
# =========================================================

import os
import joblib
import shap
import pandas as pd
import matplotlib

# Use non-interactive backend
# This prevents matplotlib windows from opening
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from feature_engineering import create_features


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
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


# Create reports folder if necessary
os.makedirs(
    REPORTS_DIR,
    exist_ok=True
)


# =========================================================
# LOAD SHAP COMPONENTS
# =========================================================

def load_shap_components():

    """
    Load the regression preprocessor,
    trained Random Forest model and SHAP explainer.
    """

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    model = joblib.load(
        MODEL_PATH
    )

    explainer = shap.TreeExplainer(
        model
    )

    return (
        preprocessor,
        model,
        explainer
    )


# =========================================================
# GET PROCESSED FEATURE NAMES
# =========================================================

def get_feature_names(preprocessor):

    """
    Get feature names after preprocessing.

    One-hot encoded categorical variables are included.
    """

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    cleaned_names = []

    for name in feature_names:

        name = str(name)

        # Remove preprocessing prefixes
        name = name.replace(
            "categorical__",
            ""
        )

        name = name.replace(
            "remainder__",
            ""
        )

        cleaned_names.append(
            name
        )

    return cleaned_names


# =========================================================
# EXPLAIN ONE STUDENT
# =========================================================

def explain_student(student_df):

    """
    Generate a SHAP explanation for one student.

    Parameters
    ----------
    student_df : pandas.DataFrame
        One-row DataFrame containing the student's
        original input features.

    Returns
    -------
    prediction : float
        Predicted final grade.

    explanation_df : pandas.DataFrame
        Feature-level SHAP explanation.
    """

    # -----------------------------------------------------
    # Validate input
    # -----------------------------------------------------

    if not isinstance(
        student_df,
        pd.DataFrame
    ):

        raise TypeError(
            "student_df must be a pandas DataFrame."
        )

    if student_df.empty:

        raise ValueError(
            "student_df cannot be empty."
        )

    if len(student_df) != 1:

        raise ValueError(
            "explain_student() expects exactly one student."
        )

    # -----------------------------------------------------
    # Create engineered features
    # -----------------------------------------------------

    student_engineered = (
        create_features(
            student_df.copy()
        )
    )

    # -----------------------------------------------------
    # Load model components
    # -----------------------------------------------------

    (
        preprocessor,
        model,
        explainer
    ) = load_shap_components()

    # -----------------------------------------------------
    # Transform student data
    # -----------------------------------------------------

    X_processed = (
        preprocessor.transform(
            student_engineered
        )
    )

    # SHAP TreeExplainer works more reliably
    # with dense arrays for this project.
    if hasattr(
        X_processed,
        "toarray"
    ):

        X_processed = (
            X_processed.toarray()
        )

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    prediction = float(
        model.predict(
            X_processed
        )[0]
    )

    # Keep grade in valid range
    prediction = max(
        0.0,
        min(
            20.0,
            prediction
        )
    )

    # -----------------------------------------------------
    # SHAP VALUES
    # -----------------------------------------------------

    shap_result = explainer(
        X_processed
    )

    shap_values = (
        shap_result.values
    )

    # For one student, take first row
    student_shap_values = (
        shap_values[0]
    )

    # -----------------------------------------------------
    # Feature names
    # -----------------------------------------------------

    feature_names = (
        get_feature_names(
            preprocessor
        )
    )

    # Safety check
    if len(feature_names) != len(
        student_shap_values
    ):

        raise ValueError(
            "Number of feature names does not "
            "match number of SHAP values."
        )

    # -----------------------------------------------------
    # CREATE EXPLANATION DATAFRAME
    # -----------------------------------------------------

    explanation_df = pd.DataFrame(
        {
            "Feature": feature_names,

            "SHAP Value":
                student_shap_values
        }
    )

    # Absolute contribution
    explanation_df[
        "Absolute Impact"
    ] = (
        explanation_df[
            "SHAP Value"
        ].abs()
    )

    # Positive / negative impact
    explanation_df[
        "Impact"
    ] = explanation_df[
        "SHAP Value"
    ].apply(
        lambda value:
            "Increases prediction"
            if value > 0
            else
            "Decreases prediction"
            if value < 0
            else
            "No significant impact"
    )

    # Sort by importance
    explanation_df = (
        explanation_df
        .sort_values(
            "Absolute Impact",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    return (
        prediction,
        explanation_df
    )


# =========================================================
# GLOBAL SHAP ANALYSIS
# =========================================================

def generate_global_shap_report():

    """
    Generate global SHAP feature importance
    using the complete student dataset.
    """

    # -----------------------------------------------------
    # Load dataset
    # -----------------------------------------------------

    df = pd.read_csv(
        DATA_PATH,
        sep=";"
    )

    # Separate target
    X = df.drop(
        columns=["G3"]
    )

    # Feature engineering
    X_engineered = (
        create_features(
            X.copy()
        )
    )

    # Load model components
    (
        preprocessor,
        model,
        explainer
    ) = load_shap_components()

    # Transform
    X_processed = (
        preprocessor.transform(
            X_engineered
        )
    )

    if hasattr(
        X_processed,
        "toarray"
    ):

        X_processed = (
            X_processed.toarray()
        )

    # Feature names
    feature_names = (
        get_feature_names(
            preprocessor
        )
    )

    # SHAP values
    shap_result = explainer(
        X_processed
    )

    shap_values = (
        shap_result.values
    )

    # -----------------------------------------------------
    # GLOBAL IMPORTANCE
    # -----------------------------------------------------

    mean_absolute_shap = (
        abs(shap_values)
        .mean(
            axis=0
        )
    )

    importance_df = pd.DataFrame(
        {
            "Feature":
                feature_names,

            "Mean Absolute SHAP":
                mean_absolute_shap
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "Mean Absolute SHAP",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    # Save CSV
    importance_df.to_csv(
        os.path.join(
            REPORTS_DIR,
            "shap_global_feature_importance.csv"
        ),
        index=False
    )

    # -----------------------------------------------------
    # GLOBAL SHAP PLOT
    # -----------------------------------------------------

    plt.figure(
        figsize=(10, 8)
    )

    shap.summary_plot(
        shap_values,
        X_processed,
        feature_names=feature_names,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            REPORTS_DIR,
            "shap_global_feature_importance.png"
        ),
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    return importance_df


# =========================================================
# INDIVIDUAL STUDENT REPORT
# =========================================================

def generate_student_report(
    student_index=0
):

    """
    Generate SHAP explanation for a student
    from the original dataset.
    """

    # Load dataset
    df = pd.read_csv(
        DATA_PATH,
        sep=";"
    )

    if student_index < 0:

        raise ValueError(
            "student_index cannot be negative."
        )

    if student_index >= len(df):

        raise IndexError(
            "student_index is outside the dataset."
        )

    # Get student
    student = (
        df.drop(
            columns=["G3"]
        )
        .iloc[
            [student_index]
        ]
        .copy()
    )

    # Explain
    prediction, explanation = (
        explain_student(
            student
        )
    )

    # Save explanation
    output_path = os.path.join(
        REPORTS_DIR,
        f"student_{student_index}_shap_explanation.csv"
    )

    explanation.to_csv(
        output_path,
        index=False
    )

    # -----------------------------------------------------
    # Individual SHAP bar plot
    # -----------------------------------------------------

    top_features = (
        explanation
        .head(10)
        .sort_values(
            "SHAP Value"
        )
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.barh(
        top_features["Feature"],
        top_features["SHAP Value"]
    )

    plt.xlabel(
        "SHAP Value"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        f"SHAP Explanation - Student {student_index}"
    )

    plt.tight_layout()

    plot_path = os.path.join(
        REPORTS_DIR,
        f"student_{student_index}_shap_explanation.png"
    )

    plt.savefig(
        plot_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    return (
        prediction,
        explanation
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print(
        "=" * 60
    )

    print(
        "SHAP Explainability Report"
    )

    print(
        "=" * 60
    )

    # -----------------------------------------------------
    # Global explanation
    # -----------------------------------------------------

    print(
        "\nGenerating global SHAP explanation..."
    )

    global_importance = (
        generate_global_shap_report()
    )

    print(
        "\nTop Global Features:"
    )

    print(
        global_importance.head(10)
        .to_string(index=False)
    )

    # -----------------------------------------------------
    # Student 0
    # -----------------------------------------------------

    print(
        "\nGenerating Student 0 explanation..."
    )

    prediction, explanation = (
        generate_student_report(
            student_index=0
        )
    )

    print(
        f"\nStudent 0 predicted grade: "
        f"{prediction:.2f}"
    )

    print(
        "\nTop Student 0 Factors:"
    )

    print(
        explanation.head(10)
        .to_string(index=False)
    )

    print(
        "\nSHAP reports saved successfully."
    )

    print(
        "=" * 60
    )


# =========================================================
# RUN ONLY WHEN FILE IS EXECUTED DIRECTLY
# =========================================================

if __name__ == "__main__":

    main()