"""
============================================================
Intelligent Student Performance Prediction System
Streamlit Application
============================================================
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
DATA_DIR = BASE_DIR / "data"

sys.path.insert(0, str(SRC_DIR))

from feature_engineering import create_features, calculate_early_warning_risk
from recommendations import generate_recommendations


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

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
    unsafe_allow_html=True,
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
        classification_model,
    )


try:
    (
        regression_preprocessor,
        regression_model,
        classification_preprocessor,
        classification_model,
    ) = load_models()

    models_loaded = True
    model_error = None

except Exception as e:
    models_loaded = False
    model_error = str(e)


# ============================================================
# SESSION STATE
# ============================================================

session_defaults = {
    "predicted_score": None,
    "support_level": None,
    "recommendations": [],
    "probabilities": {},
    "risk_score": None,
    "risk_level": None,
    "risk_factors": [],
    "last_student": None,
    "last_prediction_time": None,
}

for key, default_value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def support_group(score):
    if score < 10:
        return "High Support"
    elif score < 15:
        return "Medium Support"
    return "Low Support"


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def prepare_student_dataframe(student_data):
    student_df = pd.DataFrame([student_data])
    return create_features(student_df)


def predict_student(student_data):
    """
    Run the saved regression and classification pipelines.
    Returns prediction, support level, probability dictionary.
    """
    model_df = prepare_student_dataframe(student_data)

    regression_processed = regression_preprocessor.transform(model_df)
    predicted_score = float(
        regression_model.predict(regression_processed)[0]
    )
    predicted_score = float(np.clip(predicted_score, 0, 20))

    classification_processed = classification_preprocessor.transform(model_df)
    classification_prediction = classification_model.predict(
        classification_processed
    )[0]

    predicted_support = str(classification_prediction)
    probabilities = {}

    if hasattr(classification_model, "predict_proba"):
        probability_values = classification_model.predict_proba(
            classification_processed
        )[0]

        classes = getattr(
            classification_model,
            "classes_",
            ["High Support", "Medium Support", "Low Support"],
        )

        probabilities = {
            str(label): float(probability)
            for label, probability in zip(classes, probability_values)
        }
    else:
        predicted_support = support_group(predicted_score)

    return predicted_score, predicted_support, probabilities


def run_risk_analysis(student_data, predicted_score, predicted_support):
    """
    Transparent rule-based early-warning decision-support score.
    This is not a separately trained ML risk model.
    """
    risk_score, risk_level, risk_factors = calculate_early_warning_risk(
        predicted_score,
        predicted_support,
        student_data["absences"],
        student_data["failures"],
        student_data["studytime"],
        student_data["G1"],
        student_data["G2"],
    )

    return risk_score, risk_level, risk_factors


def store_prediction(
    student_data,
    predicted_score,
    predicted_support,
    probabilities,
    recommendations,
    risk_score,
    risk_level,
    risk_factors,
):
    st.session_state.predicted_score = predicted_score
    st.session_state.support_level = predicted_support
    st.session_state.probabilities = probabilities
    st.session_state.recommendations = recommendations
    st.session_state.risk_score = risk_score
    st.session_state.risk_level = risk_level
    st.session_state.risk_factors = risk_factors
    st.session_state.last_student = student_data


def display_risk(score, level, factors):
    if score is None:
        st.info(
            "No early-warning result is available yet. "
            "Run a student prediction first."
        )
        return

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Early Warning Risk Score",
            f"{score}/100",
        )

    with col2:
        st.metric(
            "Risk Level",
            level,
        )

    if level == "High Risk":
        st.error(
            "🔴 High Risk — Immediate academic support is recommended."
        )
    elif level == "Medium Risk":
        st.warning(
            "🟡 Medium Risk — Additional academic monitoring is recommended."
        )
    else:
        st.success(
            "🟢 Low Risk — Student is currently showing a lower risk profile."
        )

    if factors:
        st.markdown("### ⚠️ Main Risk Factors")
        for factor in factors:
            st.write(f"• {factor}")
    else:
        st.success("No major rule-based risk factors were detected.")


def make_prediction_from_student(student_data):
    if not models_loaded:
        raise RuntimeError("Prediction models are not available.")

    predicted_score, predicted_support, probabilities = predict_student(
        student_data
    )

    recommendations = generate_recommendations(
        student_data,
        predicted_support,
    )

    risk_score, risk_level, risk_factors = run_risk_analysis(
        student_data,
        predicted_score,
        predicted_support,
    )

    store_prediction(
        student_data,
        predicted_score,
        predicted_support,
        probabilities,
        recommendations,
        risk_score,
        risk_level,
        risk_factors,
    )

    return (
        predicted_score,
        predicted_support,
        probabilities,
        recommendations,
        risk_score,
        risk_level,
        risk_factors,
    )


def build_student_form(key_prefix="student"):
    """Create the complete 32-input Student Performance form."""

    st.subheader("👤 Basic Information")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        school = st.selectbox(
            "School",
            ["GP", "MS"],
            key=f"{key_prefix}_school",
        )

    with col2:
        sex = st.selectbox(
            "Gender",
            ["F", "M"],
            key=f"{key_prefix}_sex",
        )

    with col3:
        age = st.number_input(
            "Age",
            min_value=15,
            max_value=22,
            value=17,
            key=f"{key_prefix}_age",
        )

    with col4:
        address = st.selectbox(
            "Address",
            ["U", "R"],
            key=f"{key_prefix}_address",
        )

    st.subheader("👨‍👩‍👧 Family Information")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        famsize = st.selectbox(
            "Family Size",
            ["LE3", "GT3"],
            key=f"{key_prefix}_famsize",
        )

    with col2:
        Pstatus = st.selectbox(
            "Parent Status",
            ["T", "A"],
            key=f"{key_prefix}_Pstatus",
        )

    with col3:
        Medu = st.number_input(
            "Mother Education",
            min_value=0,
            max_value=4,
            value=2,
            key=f"{key_prefix}_Medu",
        )

    with col4:
        Fedu = st.number_input(
            "Father Education",
            min_value=0,
            max_value=4,
            value=2,
            key=f"{key_prefix}_Fedu",
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        Mjob = st.selectbox(
            "Mother Job",
            ["teacher", "health", "services", "at_home", "other"],
            key=f"{key_prefix}_Mjob",
        )

    with col2:
        Fjob = st.selectbox(
            "Father Job",
            ["teacher", "health", "services", "at_home", "other"],
            key=f"{key_prefix}_Fjob",
        )

    with col3:
        guardian = st.selectbox(
            "Guardian",
            ["mother", "father", "other"],
            key=f"{key_prefix}_guardian",
        )

    st.subheader("📚 Academic Information")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        studytime = st.number_input(
            "Study Time",
            min_value=1,
            max_value=4,
            value=2,
            key=f"{key_prefix}_studytime",
        )

    with col2:
        failures = st.number_input(
            "Previous Failures",
            min_value=0,
            max_value=4,
            value=0,
            key=f"{key_prefix}_failures",
        )

    with col3:
        G1 = st.number_input(
            "First Period Grade (G1)",
            min_value=0,
            max_value=20,
            value=10,
            key=f"{key_prefix}_G1",
        )

    with col4:
        G2 = st.number_input(
            "Second Period Grade (G2)",
            min_value=0,
            max_value=20,
            value=10,
            key=f"{key_prefix}_G2",
        )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        absences = st.number_input(
            "Absences",
            min_value=0,
            max_value=93,
            value=5,
            key=f"{key_prefix}_absences",
        )

    with col2:
        traveltime = st.number_input(
            "Travel Time",
            min_value=1,
            max_value=4,
            value=2,
            key=f"{key_prefix}_traveltime",
        )

    with col3:
        health = st.number_input(
            "Health",
            min_value=1,
            max_value=5,
            value=3,
            key=f"{key_prefix}_health",
        )

    with col4:
        reason = st.selectbox(
            "School Choice Reason",
            ["home", "reputation", "course", "other"],
            key=f"{key_prefix}_reason",
        )

    st.subheader("🤝 Social & Support Information")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        schoolsup = st.selectbox(
            "Extra School Support",
            ["yes", "no"],
            key=f"{key_prefix}_schoolsup",
        )

    with col2:
        famsup = st.selectbox(
            "Family Educational Support",
            ["yes", "no"],
            key=f"{key_prefix}_famsup",
        )

    with col3:
        paid = st.selectbox(
            "Extra Paid Classes",
            ["yes", "no"],
            key=f"{key_prefix}_paid",
        )

    with col4:
        activities = st.selectbox(
            "Extra Activities",
            ["yes", "no"],
            key=f"{key_prefix}_activities",
        )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        nursery = st.selectbox(
            "Attended Nursery",
            ["yes", "no"],
            key=f"{key_prefix}_nursery",
        )

    with col2:
        higher = st.selectbox(
            "Wants Higher Education",
            ["yes", "no"],
            key=f"{key_prefix}_higher",
        )

    with col3:
        internet = st.selectbox(
            "Internet Access",
            ["yes", "no"],
            key=f"{key_prefix}_internet",
        )

    with col4:
        romantic = st.selectbox(
            "Romantic Relationship",
            ["yes", "no"],
            key=f"{key_prefix}_romantic",
        )

    st.subheader("🧠 Lifestyle & Wellbeing")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        famrel = st.number_input(
            "Family Relationship",
            min_value=1,
            max_value=5,
            value=4,
            key=f"{key_prefix}_famrel",
        )

    with col2:
        freetime = st.number_input(
            "Free Time",
            min_value=1,
            max_value=5,
            value=3,
            key=f"{key_prefix}_freetime",
        )

    with col3:
        goout = st.number_input(
            "Going Out",
            min_value=1,
            max_value=5,
            value=3,
            key=f"{key_prefix}_goout",
        )

    with col4:
        Walc = st.number_input(
            "Weekend Alcohol",
            min_value=1,
            max_value=5,
            value=1,
            key=f"{key_prefix}_Walc",
        )

    with col5:
        Dalc = st.number_input(
            "Weekday Alcohol",
            min_value=1,
            max_value=5,
            value=1,
            key=f"{key_prefix}_Dalc",
        )

    return {
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
        "G2": G2,
    }


def show_prediction_result():
    if st.session_state.predicted_score is None:
        st.info("Run a prediction to display results.")
        return

    predicted_score = st.session_state.predicted_score
    support_level = st.session_state.support_level
    probabilities = st.session_state.probabilities

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Predicted Final Grade",
            f"{predicted_score:.2f}/20",
        )

    with col2:
        st.metric(
            "Support Level",
            support_level,
        )

    with col3:
        st.metric(
            "Early Warning",
            st.session_state.risk_level or "N/A",
        )

    if probabilities:
        probability_df = pd.DataFrame(
            {
                "Support Level": list(probabilities.keys()),
                "Probability": [
                    value * 100
                    for value in probabilities.values()
                ],
            }
        )

        fig = px.bar(
            probability_df,
            x="Support Level",
            y="Probability",
            text="Probability",
            title="Support-Level Prediction Probability",
        )

        fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside",
        )

        fig.update_yaxes(range=[0, 100])

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.subheader("🚨 Early Warning Risk")
    display_risk(
        st.session_state.risk_score,
        st.session_state.risk_level,
        st.session_state.risk_factors,
    )

    if st.session_state.last_student:
        export_data = dict(st.session_state.last_student)
        export_data["Predicted_G3"] = predicted_score
        export_data["Support_Level"] = support_level
        export_data["Risk_Score"] = st.session_state.risk_score
        export_data["Risk_Level"] = st.session_state.risk_level

        export_df = pd.DataFrame([export_data])

        st.download_button(
            "⬇️ Download Prediction Summary",
            export_df.to_csv(index=False),
            file_name="student_prediction_summary.csv",
            mime="text/csv",
        )


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
            "🚨 Early Warning & Risk Dashboard",
            "🔮 What-If Simulator",
            "📈 Performance Trend & Forecast",
            "🔍 Explainable AI (SHAP)",
            "🧠 Interactive SHAP Insights",
            "📚 Personalized Study Planner",
            "💡 Personalized Recommendation",
            "⚠️ Anomaly Detection",
            "👨‍🏫 Faculty/Admin Dashboard",
            "🤖 Model Benchmark & Validation",
            "👥 Student Engagement Clustering",
            "📁 Batch Scoring & Admin Portal",
            "🛡️ Dataset Validation & Upload",
            "ℹ️ About & Limitations",
        ],
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
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Student Performance Prediction • Support Classification • '
    'Explainable AI • Personalized Learning • Early Warning'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# EXECUTIVE DASHBOARD
# ============================================================

if selected_module == "🏠 Executive Dashboard":

    st.markdown(
        '<div class="section-title">🏠 Executive Dashboard</div>',
        unsafe_allow_html=True,
    )

    if not models_loaded:
        st.error("Models could not be loaded.")
        st.code(model_error)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Regression R²", "81.79%")

    with col2:
        st.metric("Regression RMSE", "1.93")

    with col3:
        st.metric("Classification Accuracy", "93.67%")

    with col4:
        st.metric("ROC-AUC", "99.08%")

    st.divider()

    st.subheader("📈 Model Performance Overview")

    performance_df = pd.DataFrame(
        {
            "Model": [
                "Linear Regression",
                "Decision Tree",
                "Random Forest",
                "Gradient Boosting",
                "SVR",
                "Tuned Random Forest",
            ],
            "R²": [
                0.7318,
                0.7599,
                0.8156,
                0.7924,
                0.7991,
                0.8179,
            ],
        }
    )

    fig = px.bar(
        performance_df,
        x="Model",
        y="R²",
        title="Regression Model R² Comparison",
        text="R²",
    )

    fig.update_yaxes(range=[0, 1])
    fig.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.subheader("🔄 System Workflow")

    workflow = pd.DataFrame(
        {
            "Stage": [
                "Student Data",
                "Validation",
                "Feature Engineering",
                "ML Prediction",
                "Early Warning",
                "SHAP Explanation",
                "Recommendation",
                "Export",
            ],
            "Status": [
                "Completed",
                "Completed",
                "Completed",
                "Completed",
                "Completed",
                "Completed",
                "Completed",
                "Completed",
            ],
        }
    )

    st.dataframe(
        workflow,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# EDA
# ============================================================

elif selected_module == "📊 Exploratory Data Analysis":

    st.markdown(
        '<div class="section-title">📊 Exploratory Data Analysis</div>',
        unsafe_allow_html=True,
    )

    if df is None:
        st.error("student-mat.csv was not found.")
    else:
        st.info(
            "Interactive visualizations generated directly from "
            "the Student Performance dataset."
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Students", len(df))

        with col2:
            st.metric("Features", len(df.columns))

        with col3:
            st.metric(
                "Missing Values",
                int(df.isnull().sum().sum()),
            )

        with col4:
            st.metric(
                "Duplicates",
                int(df.duplicated().sum()),
            )

        st.divider()

        st.subheader("1️⃣ Final Grade Distribution")
        fig1 = px.histogram(
            df,
            x="G3",
            nbins=21,
            title="Final Grade (G3) Distribution",
            labels={"G3": "Final Grade"},
        )
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("2️⃣ Study Time vs Final Grade")
        fig2 = px.scatter(
            df,
            x="studytime",
            y="G3",
            title="Study Time vs Final Grade",
            labels={
                "studytime": "Study Time",
                "G3": "Final Grade (G3)",
            },
            opacity=0.75,
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("3️⃣ Previous Failures vs Final Grade")
        fig3 = px.scatter(
            df,
            x="failures",
            y="G3",
            title="Previous Failures vs Final Grade",
            labels={
                "failures": "Number of Previous Failures",
                "G3": "Final Grade (G3)",
            },
            opacity=0.75,
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("4️⃣ Absences vs Final Grade")
        fig4 = px.scatter(
            df,
            x="absences",
            y="G3",
            title="Absences vs Final Grade",
            labels={
                "absences": "Number of Absences",
                "G3": "Final Grade (G3)",
            },
            opacity=0.75,
        )
        st.plotly_chart(fig4, use_container_width=True)

        st.subheader("5️⃣ Average Final Grade by Gender")

        gender_avg = (
            df.groupby("sex")["G3"]
            .mean()
            .reset_index()
        )

        gender_avg["Gender"] = gender_avg["sex"].map(
            {"F": "Female", "M": "Male"}
        )

        fig5 = px.bar(
            gender_avg,
            x="Gender",
            y="G3",
            title="Average Final Grade by Gender",
            labels={"G3": "Average Final Grade"},
            text="G3",
        )

        fig5.update_traces(
            texttemplate="%{text:.2f}",
            textposition="outside",
        )

        st.plotly_chart(fig5, use_container_width=True)

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
                "G3": "Average Final Grade",
            },
            text="G3",
        )

        fig6.update_traces(
            texttemplate="%{text:.2f}",
            textposition="outside",
        )

        st.plotly_chart(fig6, use_container_width=True)

        st.subheader("7️⃣ Study Time Distribution")

        study_counts = (
            df["studytime"]
            .value_counts()
            .sort_index()
            .reset_index()
        )

        study_counts.columns = [
            "Study Time",
            "Students",
        ]

        fig7 = px.bar(
            study_counts,
            x="Study Time",
            y="Students",
            title="Study Time Distribution",
            text="Students",
        )

        st.plotly_chart(fig7, use_container_width=True)

        st.subheader("8️⃣ Average Final Grade by Internet Access")

        internet_avg = (
            df.groupby("internet")["G3"]
            .mean()
            .reset_index()
        )

        internet_avg["Internet Access"] = internet_avg[
            "internet"
        ].map({"yes": "Yes", "no": "No"})

        fig8 = px.bar(
            internet_avg,
            x="Internet Access",
            y="G3",
            title="Average Final Grade by Internet Access",
            labels={"G3": "Average Final Grade"},
            text="G3",
        )

        fig8.update_traces(
            texttemplate="%{text:.2f}",
            textposition="outside",
        )

        st.plotly_chart(fig8, use_container_width=True)

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
            "G3",
        ]

        correlation = df[numerical_columns].corr()

        fig_corr = px.imshow(
            correlation,
            text_auto=".2f",
            aspect="auto",
            title="Correlation Matrix",
        )

        st.plotly_chart(
            fig_corr,
            use_container_width=True,
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
        unsafe_allow_html=True,
    )

    if not models_loaded:
        st.error("Prediction models are not available.")
    else:
        st.write(
            "Enter the student's academic, social, family, and "
            "background information."
        )

        with st.form("student_prediction_form"):

            student_data = build_student_form("prediction")

            submitted = st.form_submit_button(
                "🚀 Predict Student Performance",
                use_container_width=True,
            )

        if submitted:
            try:
                (
                    predicted_score,
                    predicted_support,
                    probabilities,
                    recommendations,
                    risk_score,
                    risk_level,
                    risk_factors,
                ) = make_prediction_from_student(student_data)

                st.success("Prediction completed successfully.")

                st.session_state.last_prediction_time = pd.Timestamp.now()

            except Exception as e:
                st.error(
                    "The model could not make a prediction."
                )
                st.exception(e)

        show_prediction_result()


# ============================================================
# EARLY WARNING DASHBOARD
# ============================================================

elif selected_module == "🚨 Early Warning & Risk Dashboard":

    st.markdown(
        '<div class="section-title">🚨 Early Warning & Risk Dashboard</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "This dashboard uses a transparent rule-based early-warning "
        "score on top of the model prediction. It is decision support, "
        "not a separately trained risk model."
    )

    if st.session_state.predicted_score is None:
        st.warning(
            "No student prediction is available. "
            "Go to Student Assessment & Prediction and run a prediction first."
        )
    else:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Predicted G3",
                f"{st.session_state.predicted_score:.2f}/20",
            )

        with col2:
            st.metric(
                "Support Level",
                st.session_state.support_level,
            )

        with col3:
            st.metric(
                "Risk Score",
                f"{st.session_state.risk_score}/100",
            )

        display_risk(
            st.session_state.risk_score,
            st.session_state.risk_level,
            st.session_state.risk_factors,
        )

        st.divider()

        st.subheader("📌 Risk Interpretation")

        risk_table = pd.DataFrame(
            {
                "Risk Level": [
                    "Low Risk",
                    "Medium Risk",
                    "High Risk",
                ],
                "Score Range": [
                    "0–29",
                    "30–59",
                    "60–100",
                ],
                "Suggested Action": [
                    "Continue monitoring",
                    "Increase academic monitoring",
                    "Immediate academic support",
                ],
            }
        )

        st.dataframe(
            risk_table,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# WHAT-IF SIMULATOR
# ============================================================

elif selected_module == "🔮 What-If Simulator":

    st.markdown(
        '<div class="section-title">🔮 What-If Simulator</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "Change selected student inputs and compare the resulting "
        "prediction with the current prediction."
    )

    if st.session_state.last_student is None:
        st.warning(
            "Run a student prediction first. "
            "The simulator uses that student as the baseline."
        )
    elif not models_loaded:
        st.error("Prediction models are not available.")
    else:
        baseline = st.session_state.last_student

        col1, col2, col3 = st.columns(3)

        with col1:
            whatif_G1 = st.slider(
                "What-if G1",
                0,
                20,
                int(baseline["G1"]),
            )

        with col2:
            whatif_G2 = st.slider(
                "What-if G2",
                0,
                20,
                int(baseline["G2"]),
            )

        with col3:
            whatif_absences = st.slider(
                "What-if Absences",
                0,
                93,
                int(baseline["absences"]),
            )

        col1, col2 = st.columns(2)

        with col1:
            whatif_studytime = st.slider(
                "What-if Study Time",
                1,
                4,
                int(baseline["studytime"]),
            )

        with col2:
            whatif_failures = st.slider(
                "What-if Previous Failures",
                0,
                4,
                int(baseline["failures"]),
            )

        whatif_student = dict(baseline)

        whatif_student["G1"] = whatif_G1
        whatif_student["G2"] = whatif_G2
        whatif_student["absences"] = whatif_absences
        whatif_student["studytime"] = whatif_studytime
        whatif_student["failures"] = whatif_failures

        try:
            baseline_score = st.session_state.predicted_score

            whatif_score, whatif_support, whatif_probabilities = predict_student(
                whatif_student
            )

            score_change = whatif_score - baseline_score

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    "Current Prediction",
                    f"{baseline_score:.2f}/20",
                )

            with c2:
                st.metric(
                    "What-If Prediction",
                    f"{whatif_score:.2f}/20",
                )

            with c3:
                st.metric(
                    "Change",
                    f"{score_change:+.2f}",
                )

            comparison_df = pd.DataFrame(
                {
                    "Scenario": [
                        "Current",
                        "What-If",
                    ],
                    "Predicted G3": [
                        baseline_score,
                        whatif_score,
                    ],
                }
            )

            fig = px.bar(
                comparison_df,
                x="Scenario",
                y="Predicted G3",
                text="Predicted G3",
                title="Current vs What-If Prediction",
            )

            fig.update_yaxes(range=[0, 20])

            fig.update_traces(
                texttemplate="%{text:.2f}",
                textposition="outside",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            st.write(
                f"**What-If Support Level:** {whatif_support}"
            )

        except Exception as e:
            st.error("What-If prediction failed.")
            st.exception(e)


# ============================================================
# PERFORMANCE TREND & FORECAST
# ============================================================

elif selected_module == "📈 Performance Trend & Forecast":

    st.markdown(
        '<div class="section-title">📈 Performance Trend & Forecast</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "This view combines G1, G2, and the current model prediction. "
        "The projected line is an illustrative trend, not a separately "
        "trained time-series forecast."
    )

    if st.session_state.last_student is None:
        st.warning(
            "Run a student prediction first."
        )
    else:
        student = st.session_state.last_student
        predicted_score = st.session_state.predicted_score

        trend_df = pd.DataFrame(
            {
                "Assessment": [
                    "G1",
                    "G2",
                    "Predicted G3",
                ],
                "Score": [
                    student["G1"],
                    student["G2"],
                    predicted_score,
                ],
            }
        )

        fig = px.line(
            trend_df,
            x="Assessment",
            y="Score",
            markers=True,
            text="Score",
            title="Student Performance Trend",
        )

        fig.update_yaxes(range=[0, 20])

        fig.update_traces(
            texttemplate="%{text:.2f}",
            textposition="top center",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        trend_change = predicted_score - student["G2"]

        if trend_change > 0:
            st.success(
                f"Projected improvement from G2 to predicted G3: "
                f"+{trend_change:.2f}"
            )
        elif trend_change < 0:
            st.warning(
                f"Projected decrease from G2 to predicted G3: "
                f"{trend_change:.2f}"
            )
        else:
            st.info("Projected G3 is equal to G2.")


# ============================================================
# SHAP EXPLAINABILITY
# ============================================================

elif selected_module == "🔍 Explainable AI (SHAP)":

    st.markdown(
        '<div class="section-title">🔍 Explainable AI (SHAP)</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "SHAP explains model contribution patterns. "
        "It does not prove causation."
    )

    possible_global_files = [
        "global_shap_summary.png",
        "shap_summary.png",
        "global_feature_importance.png",
    ]

    global_file = None

    for filename in possible_global_files:
        candidate = REPORTS_DIR / filename
        if candidate.exists():
            global_file = candidate
            break

    if global_file:
        st.subheader("🌍 Global SHAP Explanation")
        st.image(
            str(global_file),
            use_container_width=True,
        )
    else:
        st.warning(
            "Global SHAP image was not found in the reports folder."
        )

    for student_number in [0, 1, 2]:
        st.subheader(
            f"👨‍🎓 Student {student_number} SHAP Explanation"
        )

        image_candidates = [
            REPORTS_DIR / f"student_{student_number}_shap.png",
            REPORTS_DIR / f"student_{student_number}_shap_explanation.png",
            REPORTS_DIR / f"student_{student_number}_classification_shap.png",
        ]

        found_image = None

        for candidate in image_candidates:
            if candidate.exists():
                found_image = candidate
                break

        if found_image:
            st.image(
                str(found_image),
                use_container_width=True,
            )
        else:
            st.caption(
                f"Student {student_number} SHAP image not found."
            )

    st.subheader("📄 Available SHAP Reports")

    shap_files = [
        path.name
        for path in REPORTS_DIR.glob("*")
        if "shap" in path.name.lower()
    ]

    if shap_files:
        for filename in shap_files:
            st.write(f"• {filename}")
    else:
        st.info("No SHAP report files were found.")


# ============================================================
# INTERACTIVE SHAP INSIGHTS
# ============================================================

elif selected_module == "🧠 Interactive SHAP Insights":

    st.markdown(
        '<div class="section-title">🧠 Interactive SHAP Insights</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "Use this page to inspect the generated SHAP explanation files "
        "and the most important known model contributors."
    )

    shap_summary = pd.DataFrame(
        {
            "Feature": [
                "G2",
                "previous_grade_avg",
                "G1",
                "absences",
                "age",
                "assessment_trend",
                "freetime",
                "Walc",
                "schoolsup",
                "sex",
            ],
            "Role": [
                "Recent academic performance",
                "Previous grade average",
                "Earlier academic performance",
                "Attendance indicator",
                "Student age",
                "Recent grade direction",
                "Free-time pattern",
                "Weekend lifestyle indicator",
                "School support",
                "Gender feature",
            ],
        }
    )

    st.dataframe(
        shap_summary,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "These are model contribution patterns from the generated SHAP analysis, "
        "not causal relationships."
    )


# ============================================================
# PERSONALIZED STUDY PLANNER
# ============================================================

elif selected_module == "📚 Personalized Study Planner":

    st.markdown(
        '<div class="section-title">📚 Personalized Study Planner</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.last_student is None:
        st.warning(
            "Run a prediction first to generate a personalized plan."
        )
    else:
        student = st.session_state.last_student
        support = st.session_state.support_level
        risk = st.session_state.risk_level

        study_hours = {
            1: 1,
            2: 2,
            3: 3,
            4: 4,
        }.get(int(student["studytime"]), 2)

        st.subheader("🎯 Student Profile")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Support Level",
                support,
            )

        with c2:
            st.metric(
                "Risk Level",
                risk,
            )

        with c3:
            st.metric(
                "Recommended Daily Study",
                f"{study_hours} hr",
            )

        st.subheader("🗓️ Suggested Weekly Plan")

        planner = pd.DataFrame(
            {
                "Day": [
                    "Monday",
                    "Tuesday",
                    "Wednesday",
                    "Thursday",
                    "Friday",
                    "Saturday",
                    "Sunday",
                ],
                "Plan": [
                    "Review previous lessons",
                    "Practice weak topics",
                    "Solve exercises",
                    "Revision + self-test",
                    "Practice difficult questions",
                    "Weekly mock test",
                    "Review mistakes + plan next week",
                ],
                "Suggested Time": [
                    f"{study_hours} hr",
                    f"{study_hours} hr",
                    f"{study_hours} hr",
                    f"{study_hours} hr",
                    f"{study_hours} hr",
                    f"{study_hours + 1} hr",
                    "1 hr",
                ],
            }
        )

        st.dataframe(
            planner,
            use_container_width=True,
            hide_index=True,
        )

        if student["absences"] > 10:
            st.warning(
                "Attendance should be monitored because the current absence "
                "level is relatively high."
            )

        if student["G2"] < student["G1"]:
            st.warning(
                "The recent grade is decreasing. Add additional revision "
                "and practice sessions."
            )


# ============================================================
# PERSONALIZED RECOMMENDATIONS
# ============================================================

elif selected_module == "💡 Personalized Recommendation":

    st.markdown(
        '<div class="section-title">💡 Personalized Recommendation</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.predicted_score is None:
        st.warning(
            "Run a student prediction first."
        )
    else:
        st.metric(
            "Predicted Final Grade",
            f"{st.session_state.predicted_score:.2f}/20",
        )

        st.metric(
            "Support Level",
            st.session_state.support_level,
        )

        recommendations = st.session_state.recommendations

        if recommendations:
            st.subheader("🎯 Recommended Actions")

            for index, recommendation in enumerate(
                recommendations,
                start=1,
            ):
                st.write(
                    f"**{index}.** {recommendation}"
                )
        else:
            st.info(
                "No recommendation messages are available."
            )


# ============================================================
# ANOMALY DETECTION
# ============================================================

elif selected_module == "⚠️ Anomaly Detection":

    st.markdown(
        '<div class="section-title">⚠️ Anomaly Detection</div>',
        unsafe_allow_html=True,
    )

    if df is None:
        st.error("Dataset is not available.")
    else:
        from sklearn.ensemble import IsolationForest

        anomaly_columns = [
            "age",
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
            "G3",
        ]

        anomaly_data = df[anomaly_columns].copy()

        detector = IsolationForest(
            contamination=0.05,
            random_state=42,
        )

        labels = detector.fit_predict(anomaly_data)

        anomaly_result = df.copy()
        anomaly_result["Anomaly"] = np.where(
            labels == -1,
            "Potential Anomaly",
            "Normal",
        )

        anomaly_count = int(
            (anomaly_result["Anomaly"] == "Potential Anomaly").sum()
        )

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "Potential Anomalies",
                anomaly_count,
            )

        with c2:
            st.metric(
                "Normal Records",
                len(anomaly_result) - anomaly_count,
            )

        st.dataframe(
            anomaly_result[
                anomaly_result["Anomaly"] == "Potential Anomaly"
            ][
                anomaly_columns + ["Anomaly"]
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Anomaly detection flags unusual combinations of numeric "
            "features. It does not mean the student is at risk."
        )


# ============================================================
# FACULTY / ADMIN DASHBOARD
# ============================================================

elif selected_module == "👨‍🏫 Faculty/Admin Dashboard":

    st.markdown(
        '<div class="section-title">👨‍🏫 Faculty/Admin Dashboard</div>',
        unsafe_allow_html=True,
    )

    if df is None:
        st.error("Dataset is not available.")
    else:
        st.subheader("📊 Dataset Overview")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric("Students", len(df))

        with c2:
            st.metric("Average G3", f"{df['G3'].mean():.2f}")

        with c3:
            st.metric(
                "Average Absences",
                f"{df['absences'].mean():.2f}",
            )

        with c4:
            st.metric(
                "Students With Failures",
                int((df["failures"] > 0).sum()),
            )

        st.subheader("📈 Grade Distribution")

        grade_summary = (
            df["G3"]
            .value_counts()
            .sort_index()
            .reset_index()
        )

        grade_summary.columns = [
            "Grade",
            "Students",
        ]

        fig_grade = px.bar(
            grade_summary,
            x="Grade",
            y="Students",
            title="Final Grade Distribution",
        )

        st.plotly_chart(
            fig_grade,
            use_container_width=True,
        )

        st.subheader("🎯 Support-Level Distribution")

        support_series = df["G3"].apply(support_group)

        support_counts = (
            support_series
            .value_counts()
            .reindex(
                [
                    "High Support",
                    "Medium Support",
                    "Low Support",
                ],
                fill_value=0,
            )
            .reset_index()
        )

        support_counts.columns = [
            "Support Level",
            "Students",
        ]

        fig_support = px.pie(
            support_counts,
            names="Support Level",
            values="Students",
            title="Support-Level Distribution",
        )

        st.plotly_chart(
            fig_support,
            use_container_width=True,
        )

        st.subheader("🏫 School Comparison")

        school_summary = (
            df.groupby("school")
            .agg(
                Average_G3=("G3", "mean"),
                Average_Absences=("absences", "mean"),
                Students=("G3", "count"),
            )
            .reset_index()
        )

        st.dataframe(
            school_summary,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# MODEL BENCHMARK & VALIDATION
# ============================================================

elif selected_module == "🤖 Model Benchmark & Validation":

    st.markdown(
        '<div class="section-title">🤖 Model Benchmark & Validation</div>',
        unsafe_allow_html=True,
    )

    st.subheader("📈 Regression Comparison")

    regression_results = pd.DataFrame(
        {
            "Model": [
                "Mean Baseline",
                "Linear Regression",
                "Decision Tree",
                "Random Forest",
                "Gradient Boosting",
                "SVR",
                "Tuned Random Forest",
            ],
            "MAE": [
                3.6459,
                1.6373,
                1.3148,
                1.1635,
                1.2340,
                1.1910,
                1.1674,
            ],
            "RMSE": [
                4.5502,
                2.3453,
                2.2186,
                1.9446,
                2.0633,
                2.0297,
                1.9326,
            ],
            "R²": [
                -0.0097,
                0.7318,
                0.7599,
                0.8156,
                0.7924,
                0.7991,
                0.8179,
            ],
        }
    )

    st.dataframe(
        regression_results,
        use_container_width=True,
        hide_index=True,
    )

    fig_reg = px.bar(
        regression_results,
        x="Model",
        y="R²",
        title="Regression R² Comparison",
        text="R²",
    )

    fig_reg.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside",
    )

    st.plotly_chart(
        fig_reg,
        use_container_width=True,
    )

    st.subheader("🎯 Classification Comparison")

    classification_results = pd.DataFrame(
        {
            "Model": [
                "Majority Baseline",
                "Logistic Regression",
                "Decision Tree",
                "Random Forest",
                "Gradient Boosting",
                "SVM",
            ],
            "Accuracy": [
                0.4810,
                0.9241,
                0.8481,
                0.8987,
                0.8987,
                0.9367,
            ],
            "Precision": [
                0.2314,
                0.9254,
                0.8509,
                0.9032,
                0.9006,
                0.9391,
            ],
            "Recall": [
                0.4810,
                0.9241,
                0.8481,
                0.8987,
                0.8987,
                0.9367,
            ],
            "F1": [
                0.3125,
                0.9240,
                0.8492,
                0.8996,
                0.8994,
                0.9366,
            ],
        }
    )

    st.dataframe(
        classification_results,
        use_container_width=True,
        hide_index=True,
    )

    fig_cls = px.bar(
        classification_results,
        x="Model",
        y="Accuracy",
        title="Classification Accuracy Comparison",
        text="Accuracy",
    )

    fig_cls.update_yaxes(range=[0, 1])
    fig_cls.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside",
    )

    st.plotly_chart(
        fig_cls,
        use_container_width=True,
    )

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "Final Regression R²",
            "0.8179",
        )

    with c2:
        st.metric(
            "Weighted ROC-AUC",
            "0.9908",
        )

    st.subheader("🧩 SVM Confusion Matrix")

    confusion_matrix = np.array(
        [
            [25, 1, 0],
            [3, 34, 1],
            [0, 0, 15],
        ]
    )

    fig_cm = go.Figure(
        data=go.Heatmap(
            z=confusion_matrix,
            x=["High", "Medium", "Low"],
            y=["High", "Medium", "Low"],
            text=confusion_matrix,
            texttemplate="%{text}",
            colorscale="Blues",
        )
    )

    fig_cm.update_layout(
        title="SVM Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
    )

    st.plotly_chart(
        fig_cm,
        use_container_width=True,
    )

    st.subheader("🛡️ Robustness Testing")

    robustness_path = REPORTS_DIR / "robustness_test_results.csv"

    if robustness_path.exists():
        robustness_df = pd.read_csv(robustness_path)
        st.dataframe(
            robustness_df,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "robustness_test_results.csv was not found."
        )

    st.caption(
        "Robustness results are targeted perturbation tests and are "
        "not proof of universal robustness."
    )


# ============================================================
# STUDENT ENGAGEMENT CLUSTERING
# ============================================================

elif selected_module == "👥 Student Engagement Clustering":

    st.markdown(
        '<div class="section-title">👥 Student Engagement Clustering</div>',
        unsafe_allow_html=True,
    )

    if df is None:
        st.error("Dataset is not available.")
    else:
        st.info(
            "K-Means clustering is descriptive. G3 and Support_Level "
            "are not used as clustering inputs."
        )

        cluster_profiles = pd.DataFrame(
            {
                "Cluster": [0, 1, 2, 3],
                "Students": [20, 50, 204, 121],
                "Average Age": [17.85, 15.84, 16.55, 17.10],
                "Study Time": [1.40, 2.14, 2.19, 1.84],
                "Failures": [1.30, 0.28, 0.11, 0.58],
                "Absences": [7.65, 6.26, 5.89, 4.85],
            }
        )

        st.dataframe(
            cluster_profiles,
            use_container_width=True,
            hide_index=True,
        )

        fig_cluster = px.scatter(
            cluster_profiles,
            x="Study Time",
            y="Absences",
            size="Students",
            color="Cluster",
            text="Cluster",
            title="K-Means Cluster Profiles",
        )

        st.plotly_chart(
            fig_cluster,
            use_container_width=True,
        )

        st.caption(
            "Clusters are descriptive groups, not definitive risk labels."
        )


# ============================================================
# BATCH SCORING & ADMIN PORTAL
# ============================================================

elif selected_module == "📁 Batch Scoring & Admin Portal":

    st.markdown(
        '<div class="section-title">📁 Batch Scoring & Admin Portal</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "Upload a CSV containing the student input features. "
        "The app validates the required columns before scoring."
    )

    uploaded_file = st.file_uploader(
        "Upload Student CSV",
        type=["csv"],
    )

    if uploaded_file is not None:

        try:
            batch_df = pd.read_csv(uploaded_file)

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
                "G2",
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
                st.write(missing_columns)
            elif not models_loaded:
                st.error(
                    "Models are not available."
                )
            else:
                st.success(
                    f"Validation passed. {len(batch_df)} rows ready for scoring."
                )

                prediction_rows = []

                for _, row in batch_df.iterrows():
                    student = {
                        column: row[column]
                        for column in required_columns
                    }

                    try:
                        score, support, probabilities = predict_student(
                            student
                        )

                        risk_score, risk_level, risk_factors = (
                            run_risk_analysis(
                                student,
                                score,
                                support,
                            )
                        )

                        prediction_rows.append(
                            {
                                **student,
                                "Predicted_G3": round(score, 4),
                                "Support_Level": support,
                                "Risk_Score": risk_score,
                                "Risk_Level": risk_level,
                                "Risk_Factors": "; ".join(
                                    risk_factors
                                ),
                            }
                        )

                    except Exception as row_error:
                        prediction_rows.append(
                            {
                                **student,
                                "Predicted_G3": np.nan,
                                "Support_Level": "Prediction Error",
                                "Risk_Score": np.nan,
                                "Risk_Level": "Prediction Error",
                                "Risk_Factors": str(row_error),
                            }
                        )

                scored_df = pd.DataFrame(
                    prediction_rows
                )

                st.subheader("📋 Batch Prediction Results")

                st.dataframe(
                    scored_df,
                    use_container_width=True,
                    hide_index=True,
                )

                st.download_button(
                    "⬇️ Download Batch Predictions",
                    scored_df.to_csv(index=False),
                    file_name="batch_predictions.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(
                "Could not process the uploaded CSV."
            )
            st.exception(e)


# ============================================================
# DATASET VALIDATION & UPLOAD
# ============================================================

elif selected_module == "🛡️ Dataset Validation & Upload":

    st.markdown(
        '<div class="section-title">🛡️ Dataset Validation & Upload</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "This section checks the structure and basic quality of an uploaded "
        "student dataset. It does not retrain the saved models."
    )

    uploaded_dataset = st.file_uploader(
        "Upload Dataset for Validation",
        type=["csv"],
        key="validation_dataset",
    )

    if uploaded_dataset is None:
        if df is not None:
            validation_df = df
            st.caption(
                "Showing validation results for the currently loaded dataset."
            )
        else:
            validation_df = None
    else:
        try:
            validation_df = pd.read_csv(
                uploaded_dataset,
                sep=";" if uploaded_dataset.name.endswith(".csv") else ",",
            )
        except Exception as e:
            validation_df = None
            st.error("Could not read the uploaded dataset.")
            st.exception(e)

    if validation_df is not None:

        required_for_project = [
            "school",
            "sex",
            "age",
            "address",
            "studytime",
            "failures",
            "absences",
            "G1",
            "G2",
            "G3",
        ]

        missing = [
            column
            for column in required_for_project
            if column not in validation_df.columns
        ]

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Rows",
                len(validation_df),
            )

        with c2:
            st.metric(
                "Columns",
                len(validation_df.columns),
            )

        with c3:
            st.metric(
                "Missing Values",
                int(
                    validation_df.isnull()
                    .sum()
                    .sum()
                ),
            )

        if missing:
            st.error(
                "Missing core project columns:"
            )
            st.write(missing)
        else:
            st.success(
                "Core project columns are present."
            )

        duplicate_count = int(
            validation_df.duplicated().sum()
        )

        st.write(
            f"**Duplicate rows:** {duplicate_count}"
        )

        numeric_checks = [
            column
            for column in [
                "age",
                "studytime",
                "failures",
                "absences",
                "G1",
                "G2",
                "G3",
            ]
            if column in validation_df.columns
        ]

        if numeric_checks:
            st.subheader("📏 Numeric Range Summary")

            range_df = validation_df[
                numeric_checks
            ].describe().T[
                ["min", "max", "mean", "std"]
            ]

            st.dataframe(
                range_df,
                use_container_width=True,
            )

        st.subheader("👀 Dataset Preview")

        st.dataframe(
            validation_df.head(20),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# ABOUT & LIMITATIONS
# ============================================================

elif selected_module == "ℹ️ About & Limitations":

    st.markdown(
        '<div class="section-title">ℹ️ About & Limitations</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ### 🎓 Project

        **Intelligent Student Performance Prediction and Personalized
        Learning Recommendation System**

        The application combines:

        - Regression for final grade prediction
        - Classification for support-level prediction
        - K-Means clustering
        - SHAP explainability
        - Personalized recommendations
        - Early-warning risk analysis
        - What-if analysis
        - Batch scoring
        - Dataset validation
        - Robustness testing
        - Automated tests
        """
    )

    st.subheader("📊 Dataset")

    st.write(
        "The project uses the UCI Student Performance Mathematics dataset "
        "(`student-mat.csv`) containing 395 records and 33 original columns "
        "including G3. The workflow does not use personally identifiable information."
    )

    st.subheader("🤖 Final Models")

    st.write(
        "- Regression: Tuned Random Forest"
    )
    st.write(
        "- Classification: SVM"
    )
    st.write(
        "- Clustering: K-Means, K=4"
    )

    st.subheader("⚠️ Important Limitations")

    limitations = [
        "Predictions are decision-support signals, not definitive judgments.",
        "No claim of 100% accuracy should be made.",
        "Model performance depends on data quality and distribution.",
        "SHAP contribution patterns should not be interpreted as causation.",
        "The support categories are defined by project thresholds.",
        "The early-warning score is a transparent rule-based score, not a separately trained AI risk model.",
        "The what-if and trend views are analytical tools and should not be interpreted as guaranteed future outcomes.",
        "Real deployment would require additional validation, monitoring, privacy review, fairness assessment, and stakeholder oversight.",
    ]

    for limitation in limitations:
        st.write(f"• {limitation}")

    st.subheader("🔐 Privacy & Ethics")

    st.write(
        "The system is intended for academic demonstration and decision support. "
        "Any real deployment should use appropriately governed data, preserve privacy, "
        "keep humans in the decision loop, and avoid using predictions as automatic "
        "high-stakes decisions."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🎓 Intelligent Student Performance AI • "
    "Python • Pandas • NumPy • Scikit-learn • SHAP • Streamlit"
)

st.caption(
    "Academic decision-support prototype — predictions should be reviewed by a human educator."
)
