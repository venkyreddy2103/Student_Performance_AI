# ============================================================
# Intelligent Student Performance Prediction System
# Streamlit Application
# ============================================================

import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path

# ------------------------------------------------------------
# PATH CONFIGURATION
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
DATA_DIR = BASE_DIR / "data"

sys.path.insert(0, str(SRC_DIR))

# Existing project modules
from feature_engineering import create_features
from recommendations import generate_recommendations


# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ------------------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------------------

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #777;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 28px;
        font-weight: 650;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    .metric-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #ddd;
        background-color: #fafafa;
        text-align: center;
    }

    .success-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #eaf7ee;
        border-left: 5px solid #28a745;
        margin: 10px 0;
    }

    .warning-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #fff8e5;
        border-left: 5px solid #ffc107;
        margin: 10px 0;
    }

    .info-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #eef6ff;
        border-left: 5px solid #2196f3;
        margin: 10px 0;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_dataset():

    dataset_path = DATA_DIR / "student-mat.csv"

    if not dataset_path.exists():
        return None

    return pd.read_csv(dataset_path, sep=";")


df = load_dataset()


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    regression_preprocessor = joblib.load(
        MODELS_DIR / "regression_preprocessor.pkl"
    )

    regression_model = joblib.load(
        MODELS_DIR / "tuned_random_forest_regressor.pkl"
    )

    classification_preprocessor = joblib.load(
        MODELS_DIR / "classification_preprocessor.pkl"
    )

    classification_model = joblib.load(
        MODELS_DIR / "svm_classifier.pkl"
    )

    return (
        regression_preprocessor,
        regression_model,
        classification_preprocessor,
        classification_model
    )


try:

    (
        regression_preprocessor,
        regression_model,
        classification_preprocessor,
        classification_model
    ) = load_models()

    models_loaded = True

except Exception as e:

    models_loaded = False
    model_error = str(e)


# ============================================================
# SESSION STATE
# ============================================================

if "predicted_score" not in st.session_state:
    st.session_state.predicted_score = None

if "support_level" not in st.session_state:
    st.session_state.support_level = None

if "recommendations" not in st.session_state:
    st.session_state.recommendations = []

if "probabilities" not in st.session_state:
    st.session_state.probabilities = {}


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    st.markdown("## 🎓 Student Performance AI")

    st.caption("Intelligent Student Performance Prediction")

    st.divider()

    selected_module = st.radio(
        "Navigation Center",
        [
            "🏠 Executive Dashboard",
            "📊 Exploratory Data Analysis",
            "🎓 Student Assessment & Prediction",
            "🔍 Explainable AI (SHAP)",
            "💡 Personalized Recommendation",
            "🤖 Model Benchmark & Validation",
            "👥 Student Engagement Clustering",
            "📁 Batch Scoring & Admin Portal",
            "ℹ️ About & Limitations"
        ]
    )

    st.divider()

    st.markdown("### ⚡ System Telemetry")

    if models_loaded:
        st.success("Models: Online")
    else:
        st.error("Models: Error")

    if df is not None:
        st.success("Dataset: Loaded")
        st.caption(f"Records: {len(df)}")
    else:
        st.error("Dataset: Missing")


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🎓 Intelligent Student Performance AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Student Performance Prediction • Support Classification • '
    'Explainable AI • Personalized Learning'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# EXECUTIVE DASHBOARD
# ============================================================

if selected_module == "🏠 Executive Dashboard":

    st.markdown(
        '<div class="section-title">🏠 Executive Dashboard</div>',
        unsafe_allow_html=True
    )

    if not models_loaded:
        st.error("Models could not be loaded.")
        st.code(model_error)

    # --------------------------------------------------------
    # KEY METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Regression R²",
            "81.79%"
        )

    with col2:
        st.metric(
            "Regression RMSE",
            "1.93"
        )

    with col3:
        st.metric(
            "Classification Accuracy",
            "93.67%"
        )

    with col4:
        st.metric(
            "ROC-AUC",
            "99.08%"
        )

    st.divider()

    # --------------------------------------------------------
    # MODEL PERFORMANCE CHART
    # --------------------------------------------------------

    st.subheader("📈 Model Performance Overview")

    performance_df = pd.DataFrame(
        {
            "Model": [
                "Linear Regression",
                "Decision Tree",
                "Random Forest",
                "Gradient Boosting",
                "SVR",
                "Tuned Random Forest"
            ],
            "R²": [
                0.7318,
                0.7599,
                0.8156,
                0.7924,
                0.7991,
                0.8179
            ]
        }
    )

    fig = px.bar(
        performance_df,
        x="Model",
        y="R²",
        title="Regression Model R² Comparison",
        text="R²"
    )

    fig.update_yaxes(range=[0, 1])

    fig.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # WORKFLOW
    # --------------------------------------------------------

    st.subheader("🔄 System Workflow")

    workflow = pd.DataFrame(
        {
            "Stage": [
                "Student Data",
                "Validation",
                "Feature Engineering",
                "ML Prediction",
                "SHAP Explanation",
                "Recommendation"
            ],
            "Status": [
                "Completed",
                "Completed",
                "Completed",
                "Completed",
                "Completed",
                "Completed"
            ]
        }
    )

    st.dataframe(
        workflow,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================

elif selected_module == "📊 Exploratory Data Analysis":

    st.markdown(
        '<div class="section-title">📊 Exploratory Data Analysis</div>',
        unsafe_allow_html=True
    )

    if df is None:

        st.error("student-mat.csv was not found.")

    else:

        st.info(
            "Interactive visualizations generated directly from the "
            "Student Performance dataset."
        )

        # ----------------------------------------------------
        # DATASET SUMMARY
        # ----------------------------------------------------

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Students",
                len(df)
            )

        with col2:
            st.metric(
                "Features",
                len(df.columns)
            )

        with col3:
            st.metric(
                "Missing Values",
                int(df.isnull().sum().sum())
            )

        with col4:
            st.metric(
                "Duplicates",
                int(df.duplicated().sum())
            )

        st.divider()

        # ====================================================
        # GRAPH 1 - FINAL GRADE DISTRIBUTION
        # ====================================================

        st.subheader("1️⃣ Final Grade Distribution")

        fig1 = px.histogram(
            df,
            x="G3",
            nbins=21,
            title="Final Grade (G3) Distribution",
            labels={
                "G3": "Final Grade"
            }
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

        # ====================================================
        # GRAPH 2 - STUDY TIME VS FINAL GRADE
        # ====================================================

        st.subheader("2️⃣ Study Time vs Final Grade")

        fig2 = px.scatter(
            df,
            x="studytime",
            y="G3",
            title="Study Time vs Final Grade",
            labels={
                "studytime": "Study Time",
                "G3": "Final Grade (G3)"
            },
            opacity=0.75
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

        # ====================================================
        # GRAPH 3 - PREVIOUS FAILURES VS FINAL GRADE
        # ====================================================

        st.subheader("3️⃣ Previous Failures vs Final Grade")

        fig3 = px.scatter(
            df,
            x="failures",
            y="G3",
            title="Previous Failures vs Final Grade",
            labels={
                "failures": "Number of Previous Failures",
                "G3": "Final Grade (G3)"
            },
            opacity=0.75
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

        # ====================================================
        # GRAPH 4 - ABSENCES VS FINAL GRADE
        # ====================================================

        st.subheader("4️⃣ Absences vs Final Grade")

        fig4 = px.scatter(
            df,
            x="absences",
            y="G3",
            title="Absences vs Final Grade",
            labels={
                "absences": "Number of Absences",
                "G3": "Final Grade (G3)"
            },
            opacity=0.75
        )

        st.plotly_chart(
            fig4,
            use_container_width=True
        )

        # ====================================================
        # GRAPH 5 - GENDER VS FINAL GRADE
        # ====================================================

        st.subheader("5️⃣ Average Final Grade by Gender")

        gender_avg = (
            df.groupby("sex")["G3"]
            .mean()
            .reset_index()
        )

        gender_avg["Gender"] = gender_avg["sex"].map(
            {
                "F": "Female",
                "M": "Male"
            }
        )

        fig5 = px.bar(
            gender_avg,
            x="Gender",
            y="G3",
            title="Average Final Grade by Gender",
            labels={
                "G3": "Average Final Grade"
            },
            text="G3"
        )

        fig5.update_traces(
            texttemplate="%{text:.2f}",
            textposition="outside"
        )

        st.plotly_chart(
            fig5,
            use_container_width=True
        )

        # ====================================================
        # GRAPH 6 - SCHOOL VS FINAL GRADE
        # ====================================================

        st.subheader("6️⃣ Average Final Grade by School")

        school_avg = (
            df.groupby("school")["G3"]
            .mean()
            .reset_index()
        )

        fig6 = px.bar(
            school_avg,
            x="school",
            y="G3",
            title="Average Final Grade by School",
            labels={
                "school": "School",
                "G3": "Average Final Grade"
            },
            text="G3"
        )

        fig6.update_traces(
            texttemplate="%{text:.2f}",
            textposition="outside"
        )

        st.plotly_chart(
            fig6,
            use_container_width=True
        )

        # ====================================================
        # GRAPH 7 - STUDY TIME DISTRIBUTION
        # ====================================================

        st.subheader("7️⃣ Study Time Distribution")

        study_counts = (
            df["studytime"]
            .value_counts()
            .sort_index()
            .reset_index()
        )

        study_counts.columns = [
            "Study Time",
            "Students"
        ]

        fig7 = px.bar(
            study_counts,
            x="Study Time",
            y="Students",
            title="Study Time Distribution",
            text="Students"
        )

        st.plotly_chart(
            fig7,
            use_container_width=True
        )

        # ====================================================
        # GRAPH 8 - INTERNET ACCESS VS FINAL GRADE
        # ====================================================

        st.subheader("8️⃣ Average Final Grade by Internet Access")

        internet_avg = (
            df.groupby("internet")["G3"]
            .mean()
            .reset_index()
        )

        internet_avg["Internet Access"] = internet_avg[
            "internet"
        ].map(
            {
                "yes": "Yes",
                "no": "No"
            }
        )

        fig8 = px.bar(
            internet_avg,
            x="Internet Access",
            y="G3",
            title="Average Final Grade by Internet Access",
            labels={
                "G3": "Average Final Grade"
            },
            text="G3"
        )

        fig8.update_traces(
            texttemplate="%{text:.2f}",
            textposition="outside"
        )

        st.plotly_chart(
            fig8,
            use_container_width=True
        )

        # ====================================================
        # CORRELATION HEATMAP
        # ====================================================

        st.subheader("📌 Numerical Feature Correlation")

        numerical_columns = [
            "age",
            "Medu",
            "Fedu",
            "traveltime",
            "studytime",
            "failures",
            "famrel",
            "freetime",
            "goout",
            "Dalc",
            "Walc",
            "health",
            "absences",
            "G1",
            "G2",
            "G3"
        ]

        correlation = df[numerical_columns].corr()

        fig_corr = px.imshow(
            correlation,
            text_auto=".2f",
            aspect="auto",
            title="Correlation Matrix"
        )

        st.plotly_chart(
            fig_corr,
            use_container_width=True
        )

        st.success(
            "EDA completed: 8 required visualizations + correlation analysis."
        )


# ============================================================
# STUDENT ASSESSMENT & PREDICTION
# ============================================================

elif selected_module == "🎓 Student Assessment & Prediction":

    st.markdown(
        '<div class="section-title">🎓 Student Assessment & Prediction</div>',
        unsafe_allow_html=True
    )

    if not models_loaded:

        st.error("Prediction models are not available.")

    else:

        st.write(
            "Enter the student's academic and background information."
        )

        # ----------------------------------------------------
        # BASIC INFORMATION
        # ----------------------------------------------------

        st.subheader("👤 Basic Information")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            school = st.selectbox(
                "School",
                ["GP", "MS"]
            )

        with col2:
            sex = st.selectbox(
                "Gender",
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

        # ----------------------------------------------------
        # FAMILY INFORMATION
        # ----------------------------------------------------

        st.subheader("👨‍👩‍👧 Family Information")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            famsize = st.selectbox(
                "Family Size",
                ["LE3", "GT3"]
            )

        with col2:
            Pstatus = st.selectbox(
                "Parent Status",
                ["T", "A"]
            )

        with col3:
            Medu = st.number_input(
                "Mother Education",
                min_value=0,
                max_value=4,
                value=2
            )

        with col4:
            Fedu = st.number_input(
                "Father Education",
                min_value=0,
                max_value=4,
                value=2
            )

        col1, col2, col3 = st.columns(3)

        with col1:
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

        with col2:
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

        with col3:
            guardian = st.selectbox(
                "Guardian",
                [
                    "mother",
                    "father",
                    "other"
                ]
            )

        # ----------------------------------------------------
        # ACADEMIC INFORMATION
        # ----------------------------------------------------

        st.subheader("📚 Academic Information")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            studytime = st.number_input(
                "Study Time",
                min_value=1,
                max_value=4,
                value=2
            )

        with col2:
            failures = st.number_input(
                "Previous Failures",
                min_value=0,
                max_value=3,
                value=0
            )

        with col3:
            G1 = st.number_input(
                "First Period Grade (G1)",
                min_value=0,
                max_value=20,
                value=10
            )

        with col4:
            G2 = st.number_input(
                "Second Period Grade (G2)",
                min_value=0,
                max_value=20,
                value=10
            )

        col1, col2, col3 = st.columns(3)

        with col1:
            absences = st.number_input(
                "Absences",
                min_value=0,
                max_value=100,
                value=5
            )

        with col2:
            traveltime = st.number_input(
                "Travel Time",
                min_value=1,
                max_value=4,
                value=1
            )

        with col3:
            health = st.number_input(
                "Health",
                min_value=1,
                max_value=5,
                value=3
            )

        # ----------------------------------------------------
        # SOCIAL INFORMATION
        # ----------------------------------------------------

        st.subheader("🌐 Social & Support Information")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            schoolsup = st.selectbox(
                "School Support",
                ["yes", "no"]
            )

        with col2:
            famsup = st.selectbox(
                "Family Support",
                ["yes", "no"]
            )

        with col3:
            paid = st.selectbox(
                "Extra Paid Classes",
                ["yes", "no"]
            )

        with col4:
            activities = st.selectbox(
                "Extra Activities",
                ["yes", "no"]
            )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            nursery = st.selectbox(
                "Nursery",
                ["yes", "no"]
            )

        with col2:
            higher = st.selectbox(
                "Higher Education",
                ["yes", "no"]
            )

        with col3:
            internet = st.selectbox(
                "Internet",
                ["yes", "no"]
            )

        with col4:
            romantic = st.selectbox(
                "Romantic Relationship",
                ["yes", "no"]
            )

        # ----------------------------------------------------
        # LIFESTYLE
        # ----------------------------------------------------

        st.subheader("🏃 Lifestyle")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            famrel = st.number_input(
                "Family Relationship",
                min_value=1,
                max_value=5,
                value=4
            )

        with col2:
            freetime = st.number_input(
                "Free Time",
                min_value=1,
                max_value=5,
                value=3
            )

        with col3:
            goout = st.number_input(
                "Going Out",
                min_value=1,
                max_value=5,
                value=3
            )

        with col4:
            Walc = st.number_input(
                "Weekend Alcohol",
                min_value=1,
                max_value=5,
                value=1
            )

        Dalc = st.number_input(
            "Workday Alcohol",
            min_value=1,
            max_value=5,
            value=1
        )

        # ----------------------------------------------------
        # OTHER INFORMATION
        # ----------------------------------------------------

        reason = st.selectbox(
            "Reason for Choosing School",
            [
                "home",
                "reputation",
                "course",
                "other"
            ]
        )

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        st.divider()

        predict_button = st.button(
            "🚀 Predict Student Performance",
            type="primary",
            use_container_width=True
        )

        if predict_button:

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

            student_df = pd.DataFrame([student_data])

            # ------------------------------------------------
            # FEATURE ENGINEERING
            # ------------------------------------------------

            engineered_df = create_features(
                student_df.copy()
            )

            # ------------------------------------------------
            # REGRESSION
            # ------------------------------------------------

            try:

                X_regression = regression_preprocessor.transform(
                    engineered_df
                )

                predicted_score = regression_model.predict(
                    X_regression
                )[0]

                predicted_score = float(
                    np.clip(predicted_score, 0, 20)
                )

                st.session_state.predicted_score = predicted_score

            except Exception as e:

                st.error(
                    f"Regression prediction failed: {e}"
                )

                st.stop()

            # ------------------------------------------------
            # SUPPORT CLASSIFICATION
            # ------------------------------------------------

            probability_dict = {}

            try:

                classification_input = create_features(
                    student_df.copy()
                )

                classification_input = classification_input.drop(
                    columns=[
                        "G3",
                        "Support_Level"
                    ],
                    errors="ignore"
                )

                X_classification = (
                    classification_preprocessor.transform(
                        classification_input
                    )
                )

                predicted_support = (
                    classification_model.predict(
                        X_classification
                    )[0]
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
                        classification_model.classes_
                    )

                    probability_dict = {
                        str(class_name): float(probability)
                        for class_name, probability
                        in zip(
                            classes,
                            probabilities
                        )
                    }

                st.session_state.probabilities = (
                    probability_dict
                )

            except Exception as e:

                st.warning(
                    "Classification model could not make a prediction."
                )

                st.caption(
                    f"Technical details: {e}"
                )

                if predicted_score < 10:
                    predicted_support = "High Support"

                elif predicted_score < 15:
                    predicted_support = "Medium Support"

                else:
                    predicted_support = "Low Support"

            st.session_state.support_level = (
                predicted_support
            )

            # ------------------------------------------------
            # RECOMMENDATIONS
            # ------------------------------------------------

            try:

                recommendations = (
                    generate_recommendations(
                        student_df.iloc[0],
                        predicted_support
                    )
                )

                st.session_state.recommendations = (
                    recommendations
                )

            except Exception:

                st.session_state.recommendations = []

        # ----------------------------------------------------
        # DISPLAY RESULT
        # ----------------------------------------------------

        if st.session_state.predicted_score is not None:

            st.divider()

            st.subheader("🎯 Prediction Result")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Predicted Final Grade",
                    f"{st.session_state.predicted_score:.2f} / 20"
                )

            with col2:

                st.metric(
                    "Support Level",
                    st.session_state.support_level
                )

            # ------------------------------------------------
            # PROBABILITY CHART
            # ------------------------------------------------

            if st.session_state.probabilities:

                st.subheader(
                    "📊 Support-Level Probability"
                )

                probability_df = pd.DataFrame(
                    {
                        "Support Level":
                            list(
                                st.session_state
                                .probabilities
                                .keys()
                            ),
                        "Probability":
                            list(
                                st.session_state
                                .probabilities
                                .values()
                            )
                    }
                )

                probability_df["Percentage"] = (
                    probability_df["Probability"] * 100
                )

                fig_prob = px.bar(
                    probability_df,
                    x="Support Level",
                    y="Percentage",
                    title="Predicted Support-Level Probability",
                    text="Percentage"
                )

                fig_prob.update_traces(
                    texttemplate="%{text:.1f}%",
                    textposition="outside"
                )

                st.plotly_chart(
                    fig_prob,
                    use_container_width=True
                )

            # ------------------------------------------------
            # EXPORT RESULT
            # ------------------------------------------------

            result_df = pd.DataFrame(
                {
                    "Predicted_Final_Grade": [
                        st.session_state.predicted_score
                    ],
                    "Support_Level": [
                        st.session_state.support_level
                    ]
                }
            )

            st.download_button(
                "📥 Download Prediction",
                result_df.to_csv(
                    index=False
                ),
                "student_prediction.csv",
                "text/csv",
                use_container_width=True
            )


# ============================================================
# EXPLAINABLE AI
# ============================================================

elif selected_module == "🔍 Explainable AI (SHAP)":

    st.markdown(
        '<div class="section-title">🔍 Explainable AI – SHAP</div>',
        unsafe_allow_html=True
    )

    st.info(
        "SHAP explains which features influenced the model predictions."
    )

    # --------------------------------------------------------
    # GLOBAL SHAP
    # --------------------------------------------------------

    global_files = [
        "global_shap_summary.png",
        "shap_summary.png",
        "global_feature_importance.png"
    ]

    global_found = False

    for filename in global_files:

        path = REPORTS_DIR / filename

        if path.exists():

            st.subheader("🌍 Global Feature Importance")

            st.image(
                str(path),
                use_container_width=True
            )

            global_found = True
            break

    if not global_found:

        st.warning(
            "Global SHAP image was not found in reports/."
        )

    st.divider()

    # --------------------------------------------------------
    # INDIVIDUAL STUDENTS
    # --------------------------------------------------------

    st.subheader("👤 Individual Student Explanations")

    for student_number in [0, 1, 2]:

        st.markdown(
            f"### Student {student_number}"
        )

        possible_images = [
            REPORTS_DIR /
            f"student_{student_number}_shap_explanation.png",

            REPORTS_DIR /
            f"student_{student_number}_shap.png"
        ]

        image_found = False

        for image_path in possible_images:

            if image_path.exists():

                st.image(
                    str(image_path),
                    use_container_width=True
                )

                image_found = True
                break

        csv_path = (
            REPORTS_DIR /
            f"student_{student_number}_shap_explanation.csv"
        )

        if csv_path.exists():

            explanation_df = pd.read_csv(
                csv_path
            )

            st.dataframe(
                explanation_df.head(10),
                use_container_width=True,
                hide_index=True
            )

        if not image_found and not csv_path.exists():

            st.caption(
                f"SHAP report for Student {student_number} "
                "was not found."
            )

        st.divider()


# ============================================================
# PERSONALIZED RECOMMENDATION
# ============================================================

elif selected_module == "💡 Personalized Recommendation":

    st.markdown(
        '<div class="section-title">'
        '💡 Personalized Learning Recommendation'
        '</div>',
        unsafe_allow_html=True
    )

    if st.session_state.predicted_score is None:

        st.info(
            "First perform a student prediction from "
            "Student Assessment & Prediction."
        )

    else:

        st.subheader("🎯 Student Assessment")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Predicted Grade",
                f"{st.session_state.predicted_score:.2f}/20"
            )

        with col2:

            st.metric(
                "Support Level",
                st.session_state.support_level
            )

        st.divider()

        st.subheader(
            "📚 Recommended Learning Actions"
        )

        recommendations = (
            st.session_state.recommendations
        )

        if recommendations:

            for recommendation in recommendations:

                st.markdown(
                    f"""
                    <div class="info-box">
                    💡 {recommendation}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        else:

            st.info(
                "No recommendations available."
            )


# ============================================================
# MODEL BENCHMARK
# ============================================================

elif selected_module == "🤖 Model Benchmark & Validation":

    st.markdown(
        '<div class="section-title">'
        '🤖 Model Benchmark & Validation'
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # REGRESSION
    # --------------------------------------------------------

    st.subheader("📈 Regression Models")

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

    # RMSE CHART

    fig_rmse = px.bar(
        regression_results,
        x="Model",
        y="RMSE",
        title="Regression RMSE Comparison",
        text="RMSE"
    )

    fig_rmse.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside"
    )

    st.plotly_chart(
        fig_rmse,
        use_container_width=True
    )

    # R2 CHART

    fig_r2 = px.bar(
        regression_results,
        x="Model",
        y="R²",
        title="Regression R² Comparison",
        text="R²"
    )

    fig_r2.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside"
    )

    st.plotly_chart(
        fig_r2,
        use_container_width=True
    )

    st.divider()

    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    st.subheader("🎯 Classification Models")

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

    fig_acc = px.bar(
        classification_results,
        x="Model",
        y="Accuracy",
        title="Classification Accuracy Comparison",
        text="Accuracy"
    )

    fig_acc.update_yaxes(
        range=[0, 1]
    )

    fig_acc.update_traces(
        texttemplate="%{text:.2%}",
        textposition="outside"
    )

    st.plotly_chart(
        fig_acc,
        use_container_width=True
    )

    st.metric(
        "Weighted Multiclass ROC-AUC",
        "0.9908"
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    st.subheader("🔢 SVM Confusion Matrix")

    confusion_matrix = np.array(
        [
            [25, 1, 0],
            [3, 34, 1],
            [0, 0, 15]
        ]
    )

    labels = [
        "High Support",
        "Medium Support",
        "Low Support"
    ]

    fig_cm = px.imshow(
        confusion_matrix,
        x=labels,
        y=labels,
        text_auto=True,
        title="SVM Confusion Matrix"
    )

    fig_cm.update_layout(
        xaxis_title="Predicted",
        yaxis_title="Actual"
    )

    st.plotly_chart(
        fig_cm,
        use_container_width=True
    )

    # --------------------------------------------------------
    # ROBUSTNESS
    # --------------------------------------------------------

    st.subheader("🛡️ Robustness Testing")

    robustness_path = (
        REPORTS_DIR /
        "robustness_test_results.csv"
    )

    if robustness_path.exists():

        robustness_df = pd.read_csv(
            robustness_path
        )

        st.dataframe(
            robustness_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "Robustness result file not found."
        )


# ============================================================
# STUDENT ENGAGEMENT CLUSTERING
# ============================================================

elif selected_module == "👥 Student Engagement Clustering":

    st.markdown(
        '<div class="section-title">'
        '👥 Student Engagement Clustering'
        '</div>',
        unsafe_allow_html=True
    )

    if df is None:

        st.error("Dataset not available.")

    else:

        st.info(
            "K-Means clustering is performed without using the target "
            "variable G3 or Support_Level."
        )

        # ----------------------------------------------------
        # CLUSTER SUMMARY
        # ----------------------------------------------------

        cluster_sizes = pd.DataFrame(
            {
                "Cluster": [
                    "Cluster 0",
                    "Cluster 1",
                    "Cluster 2",
                    "Cluster 3"
                ],
                "Students": [
                    20,
                    50,
                    204,
                    121
                ]
            }
        )

        st.subheader("📊 Cluster Sizes")

        fig_cluster = px.bar(
            cluster_sizes,
            x="Cluster",
            y="Students",
            title="Student Distribution Across Clusters",
            text="Students"
        )

        st.plotly_chart(
            fig_cluster,
            use_container_width=True
        )

        # ----------------------------------------------------
        # CLUSTER PROFILE
        # ----------------------------------------------------

        st.subheader("📋 Cluster Profiles")

        cluster_profile = pd.DataFrame(
            {
                "Cluster": [
                    0,
                    1,
                    2,
                    3
                ],
                "Average Age": [
                    17.85,
                    15.84,
                    16.55,
                    17.10
                ],
                "Study Time": [
                    1.40,
                    2.14,
                    2.19,
                    1.84
                ],
                "Failures": [
                    1.30,
                    0.28,
                    0.11,
                    0.58
                ],
                "Absences": [
                    7.65,
                    6.26,
                    5.89,
                    4.85
                ]
            }
        )

        st.dataframe(
            cluster_profile,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # STUDY TIME VS ABSENCES
        # ----------------------------------------------------

        st.subheader(
            "📌 Study Time vs Absences"
        )

        cluster_display = df[
            [
                "studytime",
                "absences",
                "failures",
                "age"
            ]
        ].copy()

        fig_scatter = px.scatter(
            cluster_display,
            x="studytime",
            y="absences",
            size="failures",
            hover_data=["age"],
            title="Student Engagement Pattern",
            labels={
                "studytime": "Study Time",
                "absences": "Absences",
                "failures": "Previous Failures"
            }
        )

        st.plotly_chart(
            fig_scatter,
            use_container_width=True
        )


# ============================================================
# BATCH SCORING / ADMIN PORTAL
# ============================================================

elif selected_module == "📁 Batch Scoring & Admin Portal":

    st.markdown(
        '<div class="section-title">'
        '📁 Batch Scoring & Admin Portal'
        '</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Upload a CSV containing student records for batch prediction. "
        "The uploaded file is validated before prediction."
    )

    uploaded_file = st.file_uploader(
        "Upload Student CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            batch_df = pd.read_csv(
                uploaded_file,
                sep=None,
                engine="python"
            )

            st.subheader("📋 Uploaded Data")

            st.write(
                f"Rows: {len(batch_df)} | "
                f"Columns: {len(batch_df.columns)}"
            )

            st.dataframe(
                batch_df.head(10),
                use_container_width=True
            )

            # ------------------------------------------------
            # VALIDATION
            # ------------------------------------------------

            required_columns = [
                "school",
                "sex",
                "age",
                "address",
                "famsize",
                "Pstatus",
                "Medu",
                "Fedu",
                "Mjob",
                "Fjob",
                "reason",
                "guardian",
                "traveltime",
                "studytime",
                "failures",
                "schoolsup",
                "famsup",
                "paid",
                "activities",
                "nursery",
                "higher",
                "internet",
                "romantic",
                "famrel",
                "freetime",
                "goout",
                "Dalc",
                "Walc",
                "health",
                "absences",
                "G1",
                "G2"
            ]

            missing_columns = [
                column
                for column in required_columns
                if column not in batch_df.columns
            ]

            if missing_columns:

                st.error(
                    "Missing required columns:"
                )

                st.write(
                    missing_columns
                )

            else:

                st.success(
                    "CSV validation successful."
                )

                if st.button(
                    "🚀 Run Batch Prediction",
                    type="primary"
                ):

                    engineered_batch = create_features(
                        batch_df.copy()
                    )

                    X_batch = (
                        regression_preprocessor
                        .transform(
                            engineered_batch
                        )
                    )

                    predictions = (
                        regression_model
                        .predict(X_batch)
                    )

                    predictions = np.clip(
                        predictions,
                        0,
                        20
                    )

                    output_df = batch_df.copy()

                    output_df[
                        "Predicted_Final_Grade"
                    ] = predictions

                    output_df[
                        "Support_Level"
                    ] = pd.cut(
                        predictions,
                        bins=[
                            -np.inf,
                            10,
                            15,
                            np.inf
                        ],
                        labels=[
                            "High Support",
                            "Medium Support",
                            "Low Support"
                        ],
                        right=False
                    )

                    st.subheader(
                        "🎯 Batch Prediction Results"
                    )

                    st.dataframe(
                        output_df,
                        use_container_width=True
                    )

                    st.download_button(
                        "📥 Download Batch Results",
                        output_df.to_csv(
                            index=False
                        ),
                        "batch_prediction_results.csv",
                        "text/csv",
                        use_container_width=True
                    )

        except Exception as e:

            st.error(
                f"Could not process the uploaded CSV: {e}"
            )


# ============================================================
# ABOUT & LIMITATIONS
# ============================================================

elif selected_module == "ℹ️ About & Limitations":

    st.markdown(
        '<div class="section-title">'
        'ℹ️ About & Limitations'
        '</div>',
        unsafe_allow_html=True
    )

    st.subheader(
        "🎓 Project"
    )

    st.write(
        """
        **Intelligent Student Performance Prediction and
        Personalized Learning Recommendation System**

        This project uses Machine Learning to predict student
        final performance, classify support requirements,
        identify student engagement patterns and provide
        personalized learning recommendations.
        """
    )

    st.subheader("🧠 Machine Learning")

    st.markdown(
        """
        - Linear Regression
        - Decision Tree
        - Random Forest
        - Gradient Boosting
        - Support Vector Regression
        - Tuned Random Forest
        - Logistic Regression
        - SVM Classification
        - K-Means Clustering
        """
    )

    st.subheader("🔍 Explainable AI")

    st.write(
        """
        SHAP is used to explain the contribution of individual
        features to model predictions.
        """
    )

    st.subheader("📊 Dataset")

    st.write(
        """
        The project uses the UCI Student Performance dataset.

        Dataset:
        - 395 student records
        - 33 original columns
        - Final grade G3 as the regression target
        - No missing values
        - No duplicate rows
        """
    )

    st.subheader("⚠️ Limitations")

    st.markdown(
        """
        - Predictions are decision-support outputs, not final decisions.
        - The model does not guarantee 100% accuracy.
        - Student performance can be influenced by factors not present
          in the dataset.
        - The dataset is relatively small.
        - Support-level predictions should be reviewed by teachers/faculty.
        - Correlation does not necessarily imply causation.
        """
    )

    st.subheader("🔐 Privacy & Ethics")

    st.markdown(
        """
        - Avoid unnecessary personally identifiable information.
        - Use predictions only for educational support.
        - Do not use the system for unfair discrimination.
        - Model limitations should be communicated clearly.
        - External datasets and libraries should be properly credited.
        """
    )

    st.success(
        "System is designed as an educational decision-support tool."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🎓 Intelligent Student Performance Prediction & "
    "Personalized Learning Recommendation System | "
    "AI/ML Final-Year Project"
)