import streamlit as st
import pandas as pd
import joblib
import sys
import os

# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

SRC_DIR = os.path.join(
    BASE_DIR,
    "src"
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)

REPORTS_DIR = os.path.join(
    BASE_DIR,
    "reports"
)

sys.path.insert(
    0,
    SRC_DIR
)


# =========================================================
# IMPORT PROJECT FUNCTIONS
# =========================================================

from recommendations import generate_recommendations
from feature_engineering import create_features


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title(
    "🎓 Intelligent Student Performance AI"
)

st.caption(
    "Student performance prediction, support classification, "
    "explainability and personalized recommendations"
)


# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():

    regression_model = joblib.load(
        os.path.join(
            MODELS_DIR,
            "tuned_random_forest_regressor.pkl"
        )
    )

    regression_preprocessor = joblib.load(
        os.path.join(
            MODELS_DIR,
            "regression_preprocessor.pkl"
        )
    )

    classification_model = joblib.load(
        os.path.join(
            MODELS_DIR,
            "svm_classifier.pkl"
        )
    )

    classification_preprocessor = joblib.load(
        os.path.join(
            MODELS_DIR,
            "classification_preprocessor.pkl"
        )
    )

    return (
        regression_model,
        regression_preprocessor,
        classification_model,
        classification_preprocessor
    )


# =========================================================
# MODEL STATUS
# =========================================================

try:

    (
        regression_model,
        regression_preprocessor,
        classification_model,
        classification_preprocessor
    ) = load_models()

    model_status = True
    model_error = None

except Exception as e:

    model_status = False
    model_error = str(e)


# =========================================================
# SESSION STATE
# =========================================================

if "predicted_score" not in st.session_state:
    st.session_state.predicted_score = None

if "support_level" not in st.session_state:
    st.session_state.support_level = None

if "recommendations" not in st.session_state:
    st.session_state.recommendations = []

if "shap_explanation" not in st.session_state:
    st.session_state.shap_explanation = None


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🏠 Dashboard",
        "🎓 Student Prediction",
        "🧠 Explainability & Recommendations",
        "📊 Model Performance",
        "ℹ️ About & Limitations"
    ]
)


# =========================================================
# TAB 1 - DASHBOARD
# =========================================================

with tab1:

    st.header(
        "🏠 Project Dashboard"
    )

    # -----------------------------------------------------
    # MODEL STATUS
    # -----------------------------------------------------

    st.subheader(
        "⚙️ Model Status"
    )

    if model_status:

        st.success(
            "✅ All trained models loaded successfully."
        )

    else:

        st.error(
            "❌ Model loading failed."
        )

        st.code(
            model_error
        )

    # -----------------------------------------------------
    # PERFORMANCE SUMMARY
    # -----------------------------------------------------

    st.subheader(
        "📈 Model Performance Summary"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Regression R²",
            "0.8179"
        )

    with col2:

        st.metric(
            "Regression RMSE",
            "1.9326"
        )

    with col3:

        st.metric(
            "Classification Accuracy",
            "93.67%"
        )

    with col4:

        st.metric(
            "Weighted ROC-AUC",
            "0.9908"
        )

    # -----------------------------------------------------
    # PROJECT WORKFLOW
    # -----------------------------------------------------

    st.subheader(
        "🔄 Project Workflow"
    )

    workflow = pd.DataFrame(
        {
            "Step": [
                "1",
                "2",
                "3",
                "4",
                "5",
                "6"
            ],
            "Process": [
                "Student Input",
                "Data Preprocessing",
                "Feature Engineering",
                "ML Prediction",
                "SHAP Explanation",
                "Personalized Recommendation"
            ],
            "Status": [
                "Ready",
                "Ready",
                "Ready",
                "Ready",
                "Ready",
                "Ready"
            ]
        }
    )

    st.dataframe(
        workflow,
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------------------------------
    # DATASET UPLOAD
    # -----------------------------------------------------

    st.subheader(
        "📁 Dataset Upload & Validation"
    )

    st.info(
        "Upload a CSV file to inspect and validate it. "
        "Uploading a dataset does not automatically retrain the models."
    )

    uploaded_file = st.file_uploader(
        "Upload Student Dataset",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            uploaded_df = pd.read_csv(
                uploaded_file,
                sep=";"
            )

            st.success(
                f"Dataset loaded successfully: "
                f"{uploaded_df.shape[0]} rows × "
                f"{uploaded_df.shape[1]} columns"
            )

            st.write(
                "Preview:"
            )

            st.dataframe(
                uploaded_df.head(10),
                use_container_width=True
            )

            missing_values = (
                uploaded_df.isnull()
                .sum()
                .sum()
            )

            duplicate_rows = (
                uploaded_df.duplicated()
                .sum()
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Missing Values",
                    int(missing_values)
                )

            with col2:

                st.metric(
                    "Duplicate Rows",
                    int(duplicate_rows)
                )

        except Exception as e:

            st.error(
                f"❌ Could not read dataset: {e}"
            )


# =========================================================
# TAB 2 - STUDENT PREDICTION
# =========================================================

with tab2:

    st.header(
        "🎓 Student Performance Prediction"
    )

    if not model_status:

        st.error(
            "Models are not available. "
            "Please check the models folder."
        )

        st.stop()

    # =====================================================
    # PERSONAL INFORMATION
    # =====================================================

    st.subheader(
        "👤 Student Information"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        school = st.selectbox(
            "School",
            ["GP", "MS"]
        )

    with col2:

        sex = st.selectbox(
            "Sex",
            ["F", "M"]
        )

    with col3:

        age = st.number_input(
            "Age",
            min_value=15,
            max_value=22,
            value=17
        )

    with col4:

        address = st.selectbox(
            "Address",
            ["U", "R"]
        )

    col1, col2 = st.columns(2)

    with col1:

        famsize = st.selectbox(
            "Family Size",
            ["GT3", "LE3"]
        )

    with col2:

        Pstatus = st.selectbox(
            "Parent Status",
            ["T", "A"]
        )

    # =====================================================
    # FAMILY & EDUCATION
    # =====================================================

    st.subheader(
        "👨‍👩‍👧 Family & Education"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        Medu = st.number_input(
            "Mother Education",
            0,
            4,
            2
        )

    with col2:

        Fedu = st.number_input(
            "Father Education",
            0,
            4,
            2
        )

    with col3:

        Mjob = st.selectbox(
            "Mother Job",
            [
                "teacher",
                "health",
                "services",
                "at_home",
                "other"
            ]
        )

    with col4:

        Fjob = st.selectbox(
            "Father Job",
            [
                "teacher",
                "health",
                "services",
                "at_home",
                "other"
            ]
        )

    col1, col2 = st.columns(2)

    with col1:

        reason = st.selectbox(
            "Reason for Choosing School",
            [
                "course",
                "home",
                "reputation",
                "other"
            ]
        )

    with col2:

        guardian = st.selectbox(
            "Guardian",
            [
                "mother",
                "father",
                "other"
            ]
        )

    # =====================================================
    # TRAVEL & STUDY
    # =====================================================

    st.subheader(
        "📚 Study & Academic Habits"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        traveltime = st.number_input(
            "Travel Time",
            1,
            4,
            2
        )

    with col2:

        studytime = st.number_input(
            "Study Time",
            1,
            4,
            2
        )

    with col3:

        failures = st.number_input(
            "Previous Failures",
            0,
            3,
            0
        )

    with col4:

        schoolsup = st.selectbox(
            "School Support",
            ["yes", "no"]
        )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        famsup = st.selectbox(
            "Family Support",
            ["yes", "no"]
        )

    with col2:

        paid = st.selectbox(
            "Paid Classes",
            ["yes", "no"]
        )

    with col3:

        activities = st.selectbox(
            "Extra Activities",
            ["yes", "no"]
        )

    with col4:

        nursery = st.selectbox(
            "Nursery Education",
            ["yes", "no"]
        )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        higher = st.selectbox(
            "Wants Higher Education",
            ["yes", "no"]
        )

    with col2:

        internet = st.selectbox(
            "Internet Access",
            ["yes", "no"]
        )

    with col3:

        romantic = st.selectbox(
            "Romantic Relationship",
            ["yes", "no"]
        )

    with col4:

        famrel = st.number_input(
            "Family Relationship",
            1,
            5,
            4
        )

    # =====================================================
    # LIFESTYLE
    # =====================================================

    st.subheader(
        "🏃 Lifestyle"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        freetime = st.number_input(
            "Free Time",
            1,
            5,
            3
        )

    with col2:

        goout = st.number_input(
            "Going Out",
            1,
            5,
            3
        )

    with col3:

        health = st.number_input(
            "Health",
            1,
            5,
            3
        )

    with col4:

        Dalc = st.number_input(
            "Workday Alcohol",
            1,
            5,
            1
        )

    col1, col2 = st.columns(2)

    with col1:

        Walc = st.number_input(
            "Weekend Alcohol",
            1,
            5,
            1
        )

    with col2:

        absences = st.number_input(
            "Absences",
            0,
            100,
            5
        )

    # =====================================================
    # PREVIOUS GRADES
    # =====================================================

    st.subheader(
        "📊 Previous Grades"
    )

    col1, col2 = st.columns(2)

    with col1:

        G1 = st.number_input(
            "First Period Grade (G1)",
            0,
            20,
            10
        )

    with col2:

        G2 = st.number_input(
            "Second Period Grade (G2)",
            0,
            20,
            10
        )

    st.divider()

    # =====================================================
    # PREDICTION BUTTON
    # =====================================================

    predict_button = st.button(
        "🔮 Predict Student Performance",
        type="primary",
        use_container_width=True
    )

    # =====================================================
    # PREDICTION
    # =====================================================

    if predict_button:

        # -------------------------------------------------
        # VALIDATION
        # -------------------------------------------------

        if not (0 <= G1 <= 20):

            st.error(
                "❌ G1 must be between 0 and 20."
            )

            st.stop()

        if not (0 <= G2 <= 20):

            st.error(
                "❌ G2 must be between 0 and 20."
            )

            st.stop()

        if absences < 0:

            st.error(
                "❌ Absences cannot be negative."
            )

            st.stop()

        if not (1 <= studytime <= 4):

            st.error(
                "❌ Study time must be between 1 and 4."
            )

            st.stop()

        if not (0 <= failures <= 3):

            st.error(
                "❌ Previous failures must be between 0 and 3."
            )

            st.stop()

        # -------------------------------------------------
        # STUDENT DATA
        # -------------------------------------------------

        student_data = {

            "school": school,
            "sex": sex,
            "age": age,
            "address": address,
            "famsize": famsize,
            "Pstatus": Pstatus,

            "Medu": Medu,
            "Fedu": Fedu,
            "Mjob": Mjob,
            "Fjob": Fjob,

            "reason": reason,
            "guardian": guardian,

            "traveltime": traveltime,
            "studytime": studytime,
            "failures": failures,

            "schoolsup": schoolsup,
            "famsup": famsup,
            "paid": paid,
            "activities": activities,

            "nursery": nursery,
            "higher": higher,
            "internet": internet,
            "romantic": romantic,

            "famrel": famrel,
            "freetime": freetime,
            "goout": goout,

            "Dalc": Dalc,
            "Walc": Walc,

            "health": health,
            "absences": absences,

            "G1": G1,
            "G2": G2
        }

        student_df = pd.DataFrame(
            [student_data]
        )

        # -------------------------------------------------
        # REGRESSION
        # -------------------------------------------------

        try:

            student_engineered = create_features(
                student_df.copy()
            )

            X_student = (
                regression_preprocessor
                .transform(student_engineered)
            )

            predicted_score = float(
                regression_model
                .predict(X_student)[0]
            )

            predicted_score = max(
                0.0,
                min(
                    20.0,
                    predicted_score
                )
            )

        except Exception as e:

            st.error(
                f"❌ Regression prediction failed: {e}"
            )

            st.stop()

        # -------------------------------------------------
        # CLASSIFICATION
        # -------------------------------------------------

        probability_dict = {}

        try:

            classification_input = create_features(
               student_df.copy()
            )

            classification_input = classification_input.drop(
               columns=["G3", "Support_Level"],
               errors="ignore"
            )

            X_classification = (
    classification_preprocessor
    .transform(classification_input)
)

            predicted_support = (
                classification_model
                .predict(X_classification)[0]
            )

            if hasattr(
                classification_model,
                "predict_proba"
            ):

                probabilities = (
                    classification_model
                    .predict_proba(
                        X_classification
                    )[0]
                )

                classes = (
                    classification_model
                    .classes_
                )

                probability_dict = {

                    str(class_name):
                        float(probability)

                    for class_name, probability
                    in zip(
                        classes,
                        probabilities
                    )
                }

        except Exception as e:

            st.warning(
                "⚠️ Classification model could not "
                "make a prediction."
            )

            st.caption(
                f"Technical details: {e}"
            )

            # Fallback
            if predicted_score < 10:

                predicted_support = (
                    "High Support"
                )

            elif predicted_score < 15:

                predicted_support = (
                    "Medium Support"
                )

            else:

                predicted_support = (
                    "Low Support"
                )

        # -------------------------------------------------
        # SESSION STATE
        # -------------------------------------------------

        st.session_state.predicted_score = (
            predicted_score
        )

        st.session_state.support_level = (
            predicted_support
        )

        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        st.success(
            "✅ Prediction completed successfully!"
        )

        st.subheader(
            "🎯 Prediction Result"
        )

        result_col1, result_col2 = (
            st.columns(2)
        )

        with result_col1:

            st.metric(
                "Predicted Final Grade",
                f"{predicted_score:.2f} / 20"
            )

        with result_col2:

            st.metric(
                "Support Level",
                predicted_support
            )

        # -------------------------------------------------
        # SUPPORT MESSAGE
        # -------------------------------------------------

        if predicted_support == "High Support":

            st.error(
                "🔴 High Support: "
                "The student may need additional "
                "academic support."
            )

        elif predicted_support == "Medium Support":

            st.warning(
                "🟡 Medium Support: "
                "The student may benefit from "
                "additional guidance and regular practice."
            )

        else:

            st.success(
                "🟢 Low Support: "
                "The student is currently predicted "
                "to require less additional support."
            )

        # -------------------------------------------------
        # PROBABILITIES
        # -------------------------------------------------

        if probability_dict:

            st.subheader(
                "📊 Support-Level Probabilities"
            )

            probability_df = pd.DataFrame(
                {
                    "Support Level":
                        list(
                            probability_dict.keys()
                        ),

                    "Probability (%)":
                        [
                            round(
                                probability * 100,
                                2
                            )

                            for probability
                            in probability_dict.values()
                        ]
                }
            )

            st.dataframe(
                probability_df,
                use_container_width=True,
                hide_index=True
            )

            chart_df = (
                probability_df
                .set_index(
                    "Support Level"
                )
            )

            st.bar_chart(
                chart_df
            )

        # -------------------------------------------------
        # SHAP
        # -------------------------------------------------

        st.subheader(
            "🧠 Why did the model make this prediction?"
        )

        try:

            # Import here so that the application can
            # still start if SHAP has a separate issue.
            from shap_explainability import (
                explain_student
            )

            shap_prediction, shap_explanation = (
                explain_student(
                    student_df
                )
            )

            st.write(
                "The following factors had the "
                "largest contribution to this "
                "individual prediction."
            )

            top_factors = (
                shap_explanation
                .head(5)
            )

            for _, row in top_factors.iterrows():

                feature = str(
                    row["Feature"]
                )

                shap_value = float(
                    row["SHAP Value"]
                )

                if shap_value > 0:

                    st.success(
                        f"🟢 **{feature}** → "
                        f"increased the predicted "
                        f"grade by {shap_value:.2f}"
                    )

                elif shap_value < 0:

                    st.error(
                        f"🔴 **{feature}** → "
                        f"decreased the predicted "
                        f"grade by {abs(shap_value):.2f}"
                    )

                else:

                    st.info(
                        f"⚪ **{feature}** → "
                        "had almost no effect."
                    )

            st.session_state.shap_explanation = (
                shap_explanation
            )

        except Exception as e:

            st.warning(
                "⚠️ SHAP explanation could not "
                "be generated."
            )

            st.caption(
                f"Technical details: {e}"
            )

        # -------------------------------------------------
        # RECOMMENDATIONS
        # -------------------------------------------------

        st.subheader(
            "📚 Personalized Recommendations"
        )

        try:

            recommendations = (
                generate_recommendations(
                    student_data,
                    predicted_support
                )
            )

            st.session_state.recommendations = (
                recommendations
            )

            if recommendations:

                for recommendation in (
                    recommendations
                ):

                    st.info(
                        f"💡 {recommendation}"
                    )

            else:

                st.info(
                    "No additional recommendations "
                    "were generated."
                )

        except Exception as e:

            st.warning(
                "⚠️ Recommendations could not "
                "be generated."
            )

            st.caption(
                f"Technical details: {e}"
            )

            st.session_state.recommendations = []

        # -------------------------------------------------
        # EXPORT
        # -------------------------------------------------

        st.subheader(
            "📥 Export Prediction Summary"
        )

        summary = pd.DataFrame(
            [
                {
                    "Predicted Final Grade":
                        round(
                            predicted_score,
                            2
                        ),

                    "Support Level":
                        predicted_support,

                    "Recommendations":
                        " | ".join(
                            st.session_state
                            .recommendations
                        )
                }
            ]
        )

        csv_data = (
            summary
            .to_csv(
                index=False
            )
        )

        st.download_button(
            label="📥 Download Prediction Summary",
            data=csv_data,
            file_name=(
                "student_prediction_summary.csv"
            ),
            mime="text/csv",
            use_container_width=True
        )


# =========================================================
# TAB 3 - EXPLAINABILITY & RECOMMENDATIONS
# =========================================================

with tab3:

    st.header(
        "🧠 Explainability & Recommendations"
    )

    if (
        st.session_state.predicted_score
        is not None
    ):

        st.metric(
            "Latest Predicted Grade",
            f"{st.session_state.predicted_score:.2f} / 20"
        )

        st.metric(
            "Latest Support Level",
            st.session_state.support_level
        )

        # -------------------------------------------------
        # SHAP TABLE
        # -------------------------------------------------

        if (
            st.session_state.shap_explanation
            is not None
        ):

            st.subheader(
                "🔍 SHAP Feature Contributions"
            )

            shap_display = (
                st.session_state
                .shap_explanation
                .head(10)
            )

            st.dataframe(
                shap_display,
                use_container_width=True,
                hide_index=True
            )

        # -------------------------------------------------
        # RECOMMENDATIONS
        # -------------------------------------------------

        if (
            st.session_state.recommendations
        ):

            st.subheader(
                "📚 Recommendations"
            )

            for recommendation in (
                st.session_state
                .recommendations
            ):

                st.info(
                    f"💡 {recommendation}"
                )

    else:

        st.info(
            "Run a student prediction first "
            "to see explanations and recommendations."
        )


# =========================================================
# TAB 4 - MODEL PERFORMANCE
# =========================================================

with tab4:

    st.header(
        "📊 Model Performance"
    )

    # -----------------------------------------------------
    # REGRESSION
    # -----------------------------------------------------

    st.subheader(
        "📈 Regression Models"
    )

    regression_results = pd.DataFrame(
        {
            "Model": [
                "Mean Baseline",
                "Linear Regression",
                "Decision Tree",
                "Random Forest",
                "Gradient Boosting",
                "SVR",
                "Tuned Random Forest"
            ],

            "MAE": [
                3.6459,
                1.6373,
                1.3148,
                1.1635,
                1.2340,
                1.1910,
                1.1674
            ],

            "RMSE": [
                4.5502,
                2.3453,
                2.2186,
                1.9446,
                2.0633,
                2.0297,
                1.9326
            ],

            "R²": [
                -0.0097,
                0.7318,
                0.7599,
                0.8156,
                0.7924,
                0.7991,
                0.8179
            ]
        }
    )

    st.dataframe(
        regression_results,
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------------------------------
    # CLASSIFICATION
    # -----------------------------------------------------

    st.subheader(
        "🎯 Classification Models"
    )

    classification_results = pd.DataFrame(
        {
            "Model": [
                "Majority Baseline",
                "Logistic Regression",
                "Decision Tree",
                "Random Forest",
                "Gradient Boosting",
                "SVM"
            ],

            "Accuracy": [
                0.4810,
                0.9241,
                0.8481,
                0.8987,
                0.8987,
                0.9367
            ],

            "Precision": [
                0.2314,
                0.9254,
                0.8509,
                0.9032,
                0.9006,
                0.9391
            ],

            "Recall": [
                0.4810,
                0.9241,
                0.8481,
                0.8987,
                0.8987,
                0.9367
            ],

            "F1": [
                0.3125,
                0.9240,
                0.8492,
                0.8996,
                0.8994,
                0.9366
            ]
        }
    )

    st.dataframe(
        classification_results,
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------------------------------
    # ROC AUC
    # -----------------------------------------------------

    st.subheader(
        "📊 Classification ROC-AUC"
    )

    st.metric(
        "Weighted Multiclass ROC-AUC",
        "0.9908"
    )

    # -----------------------------------------------------
    # CONFUSION MATRIX
    # -----------------------------------------------------

    st.subheader(
        "🔲 SVM Confusion Matrix"
    )

    confusion_matrix = pd.DataFrame(
        [
            [25, 1, 0],
            [3, 34, 1],
            [0, 0, 15]
        ],

        columns=[
            "High Support",
            "Medium Support",
            "Low Support"
        ],

        index=[
            "High Support",
            "Medium Support",
            "Low Support"
        ]
    )

    st.dataframe(
        confusion_matrix,
        use_container_width=True
    )


# =========================================================
# TAB 5 - ABOUT & LIMITATIONS
# =========================================================

with tab5:

    st.header(
        "ℹ️ About the Project"
    )

    st.write(
        """
        This application predicts student final performance,
        identifies the level of academic support that may be
        appropriate, explains individual predictions using
        SHAP, and provides personalized recommendations.
        """
    )

    # -----------------------------------------------------
    # DATASET
    # -----------------------------------------------------

    st.subheader(
        "📚 Dataset"
    )

    st.write(
        """
        The project uses the UCI Student Performance dataset.
        The target variable for regression is G3, the final
        student grade.
        """
    )

    # -----------------------------------------------------
    # MACHINE LEARNING
    # -----------------------------------------------------

    st.subheader(
        "🤖 Machine Learning"
    )

    st.write(
        """
        Regression models are used to predict the final grade.

        Classification models are used to classify students
        into High Support, Medium Support and Low Support
        groups.

        K-Means clustering is also used to explore student
        groups without using the target variable.
        """
    )

    # -----------------------------------------------------
    # EXPLAINABILITY
    # -----------------------------------------------------

    st.subheader(
        "🧠 Explainability"
    )

    st.write(
        """
        SHAP is used to explain which features contributed
        positively or negatively to an individual prediction.
        """
    )

    # -----------------------------------------------------
    # LIMITATIONS
    # -----------------------------------------------------

    st.subheader(
        "⚠️ Limitations"
    )

    st.warning(
        """
        • Predictions are not guaranteed to be 100% accurate.

        • The model should be treated as a decision-support
          tool rather than a definitive judgment about a student.

        • The dataset is relatively small.

        • Previous grades G1 and G2 are highly informative
          for predicting G3.

        • Predictions may change when students provide
          different or incomplete information.

        • Model performance on this dataset does not guarantee
          the same performance on a different student population.

        • Support levels are project-defined categories and
          should not be treated as clinical or educational diagnoses.
        """
    )

    # -----------------------------------------------------
    # ETHICS
    # -----------------------------------------------------

    st.subheader(
        "🔐 Responsible AI"
    )

    st.write(
        """
        The system is intended to support educators and
        students. Predictions should be reviewed together
        with relevant academic context and should not be used
        as the sole basis for important decisions.
        """
    )

    st.success(
        "✅ The application is designed as decision-support, "
        "not as a definitive judgment of student ability."
    )