import streamlit as st
import pandas as pd
import joblib
import sys
import os

sys.path.append(os.path.abspath("src"))

from recommendations import generate_recommendations
from shap_explainability import explain_student

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Intelligent Student Performance Prediction")
st.write(
    "Predict student performance and identify the level of academic support needed."
)

# -----------------------------
# Load saved models
# -----------------------------
try:
    regression_model = joblib.load(
        "models/tuned_random_forest_regressor.pkl"
    )

    regression_preprocessor = joblib.load(
        "models/regression_preprocessor.pkl"
    )

    classification_model = joblib.load(
        "models/svm_classifier.pkl"
    )

    classification_preprocessor = joblib.load(
        "models/classification_preprocessor.pkl"
    )

    st.success("✅ Models loaded successfully")

except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    st.stop()


# -----------------------------
# Student Information
# -----------------------------
st.header("👨‍🎓 Student Information")

col1, col2, col3 = st.columns(3)

with col1:
    school = st.selectbox("School", ["GP", "MS"])
    sex = st.selectbox("Sex", ["F", "M"])
    age = st.number_input("Age", min_value=15, max_value=22, value=17)
    address = st.selectbox("Address", ["U", "R"])
    famsize = st.selectbox("Family Size", ["GT3", "LE3"])

with col2:
    Pstatus = st.selectbox("Parent Status", ["T", "A"])
    Medu = st.number_input("Mother Education", 0, 4, 2)
    Fedu = st.number_input("Father Education", 0, 4, 2)
    Mjob = st.selectbox(
        "Mother Job",
        ["teacher", "health", "services", "at_home", "other"]
    )
    Fjob = st.selectbox(
        "Father Job",
        ["teacher", "health", "services", "at_home", "other"]
    )

with col3:
    reason = st.selectbox(
        "Reason for Choosing School",
        ["course", "home", "reputation", "other"]
    )
    guardian = st.selectbox(
        "Guardian",
        ["mother", "father", "other"]
    )
    traveltime = st.number_input("Travel Time", 1, 4, 2)
    studytime = st.number_input("Study Time", 1, 4, 2)
    failures = st.number_input("Previous Failures", 0, 3, 0)


# -----------------------------
# Academic Information
# -----------------------------
st.header("📚 Academic Information")

col1, col2, col3 = st.columns(3)

with col1:
    schoolsup = st.selectbox("School Support", ["yes", "no"])
    famsup = st.selectbox("Family Support", ["yes", "no"])
    paid = st.selectbox("Paid Classes", ["yes", "no"])
    activities = st.selectbox("Extra Activities", ["yes", "no"])

with col2:
    nursery = st.selectbox("Attended Nursery", ["yes", "no"])
    higher = st.selectbox("Wants Higher Education", ["yes", "no"])
    internet = st.selectbox("Internet Access", ["yes", "no"])
    romantic = st.selectbox("Romantic Relationship", ["yes", "no"])

with col3:
    famrel = st.number_input("Family Relationship Quality", 1, 5, 4)
    freetime = st.number_input("Free Time", 1, 5, 3)
    goout = st.number_input("Going Out", 1, 5, 3)
    health = st.number_input("Health", 1, 5, 3)


# -----------------------------
# Lifestyle and Previous Grades
# -----------------------------
st.header("📊 Lifestyle & Previous Grades")

col1, col2, col3, col4 = st.columns(4)

with col1:
    Dalc = st.number_input("Workday Alcohol", 1, 5, 1)

with col2:
    Walc = st.number_input("Weekend Alcohol", 1, 5, 1)

with col3:
    absences = st.number_input("Absences", 0, 100, 5)

with col4:
    G1 = st.number_input("First Period Grade (G1)", 0, 20, 10)

G2 = st.number_input("Second Period Grade (G2)", 0, 20, 10)


# -----------------------------
# Prediction button
# -----------------------------
if st.button("🔮 Predict Student Performance", type="primary"):
    # Input validation
    if not (0 <= G1 <= 20 and 0 <= G2 <= 20):
        st.error("❌ G1 and G2 must be between 0 and 20.")
        st.stop()

    if absences < 0:
        st.error("❌ Absences cannot be negative.")
        st.stop()

    if not (1 <= studytime <= 4):
        st.error("❌ Study time must be between 1 and 4.")
        st.stop()

    if not (0 <= failures <= 3):
        st.error("❌ Previous failures must be between 0 and 3.")
        st.stop()

    student = pd.DataFrame([{
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
    }])

    # Regression prediction
    X_reg = regression_preprocessor.transform(student)
    predicted_score = regression_model.predict(X_reg)[0]

    # Classification prediction
    X_class = classification_preprocessor.transform(student)
    support_level = classification_model.predict(X_class)[0]
   # SHAP explanation
    shap_explanation = explain_student(student)

    # Clean feature names
    shap_explanation["Feature"] = shap_explanation["Feature"].str.replace(
        "remainder__", "", regex=False
    ).str.replace(
        "categorical__", "", regex=False
    )

    st.header("🔍 Why did the model make this prediction?")

    st.subheader("Top factors affecting this prediction")

    for _, row in shap_explanation.head(5).iterrows():

        feature = row["Feature"]
        value = row["SHAP Value"]

        if value > 0:
            st.success(
                f"🟢 {feature} → increased prediction by {value:.2f}"
            )
        else:
            st.error(
                f"🔴 {feature} → decreased prediction by {abs(value):.2f}"
            )

    # Personalized recommendations
    recommendations = generate_recommendations(
        student.iloc[0],
        support_level
    )

    st.header("💡 Personalized Recommendations")

    for i, recommendation in enumerate(recommendations, 1):
        st.write(f"**{i}.** {recommendation}")
        # Export prediction summary
    summary = pd.DataFrame([{
        "Predicted Final Grade": round(predicted_score, 2),
        "Support Level": support_level,
        "Recommendations": " | ".join(recommendations)
    }])

    csv_data = summary.to_csv(index=False)

    st.download_button(
        label="📥 Download Prediction Summary",
        data=csv_data,
        file_name="student_prediction_summary.csv",
        mime="text/csv"
    )

    # Prediction results
    st.header("🎯 Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Predicted Final Grade",
            f"{predicted_score:.2f} / 20"
        )

    with col2:
        st.metric(
            "Support Level",
            support_level
        )
        # -----------------------------
# Model Performance Dashboard
# -----------------------------
st.header("📈 Model Performance Dashboard")

st.subheader("Regression Model")

regression_results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
        "SVR",
        "Tuned Random Forest"
    ],
    "MAE": [
        1.6467,
        1.3412,
        1.1779,
        1.1477,
        1.2051,
        1.1770
    ],
    "RMSE": [
        2.3784,
        2.4036,
        2.0014,
        2.0147,
        2.0249,
        1.9978
    ],
    "R²": [
        0.7241,
        0.7182,
        0.8047,
        0.8020,
        0.8000,
        0.8053
    ]
})

st.dataframe(
    regression_results,
    use_container_width=True
)

st.success("🏆 Best Regression Model: Tuned Random Forest")
st.subheader("Classification Model")

classification_results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
        "SVM"
    ],
    "Accuracy": [
        0.9114,
        0.8987,
        0.8734,
        0.8987,
        0.9241
    ],
    "Precision": [
        0.9122,
        0.9004,
        0.8800,
        0.9006,
        0.9286
    ],
    "Recall": [
        0.9114,
        0.8987,
        0.8734,
        0.8987,
        0.9241
    ],
    "F1 Score": [
        0.9111,
        0.8989,
        0.8737,
        0.8994,
        0.9239
    ]
})

st.dataframe(
    classification_results,
    use_container_width=True
)

st.success("🏆 Best Classification Model: SVM")
st.subheader("📊 Classification ROC-AUC")

roc_auc = 0.9893

st.metric(
    "Weighted ROC-AUC",
    f"{roc_auc:.4f}"
)

st.success("🏆 SVM achieved a weighted ROC-AUC of 0.9893")
st.subheader("🔲 Confusion Matrix")

confusion_matrix = pd.DataFrame(
    [
        [25, 1, 0],
        [4, 33, 1],
        [0, 0, 15]
    ],
    index=["High Support", "Medium Support", "Low Support"],
    columns=["High Support", "Medium Support", "Low Support"]
)

st.dataframe(
    confusion_matrix,
    use_container_width=True
)
st.subheader("📈 Confusion Matrix Visualization")

import plotly.express as px

fig = px.imshow(
    confusion_matrix,
    text_auto=True,
    labels=dict(
        x="Predicted",
        y="Actual",
        color="Number of Students"
    ),
    title="SVM Confusion Matrix"
)

st.plotly_chart(fig, use_container_width=True)