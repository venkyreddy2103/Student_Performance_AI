🎓 Student Performance AI

Machine Learning Based Student Performance Analysis & Early Support System

Student Performance AI is a Machine Learning project developed to analyze student academic performance, predict final grades, classify students according to support level, identify student groups, provide SHAP-based explanations, and generate personalized recommendations through a Streamlit application.

---

👨‍🎓 Student Information

Detail| Information
Name| Venky
Roll Number| 232G1A0489
Project| Student Performance AI
Domain| Machine Learning / Artificial Intelligence
Application| Streamlit

---

📌 Project Overview

The project uses the Student Performance dataset ("student-mat.csv") to perform different Machine Learning tasks.

The project includes:

- 📈 Student final-grade prediction
- 🎯 Support-level classification
- 👥 Student clustering
- ⚠️ Early-warning analysis
- 🧠 SHAP-based explanation
- 💡 Personalized recommendations
- 🌐 Streamlit application

---

📊 Dataset

The project uses:

data/student-mat.csv

The final grade used for the main regression task is:

G3

The dataset is processed before being supplied to the Machine Learning models.

---

📈 1. Final Grade Prediction

The regression task predicts the student's final grade G3.

The project evaluated different regression approaches and performed hyperparameter tuning.

Final Tuned Regression Result

Metric| Result
MAE| 1.17698
RMSE| 1.99784
R² Score| 0.80535

The tuned Random Forest configuration selected during hyperparameter search was:

n_estimators = 300
min_samples_split = 2
min_samples_leaf = 1
max_features = 1.0
max_depth = 10

---

🎯 2. Support-Level Classification

Students are classified into three support levels based on "G3".

Condition| Support Level
"G3 < 10"| 🔴 High Support
"G3 < 15"| 🟡 Medium Support
Otherwise| 🟢 Low Support

Dataset Class Distribution

Support Level| Students
Medium Support| 192
High Support| 130
Low Support| 73

Several classification models were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- SVM

🏆 Best Classification Result

The best result was obtained using SVM.

Metric| Result
Accuracy| 92.41%
Precision| 92.86%
F1 Score| 92.39%

Confusion Matrix

[[25, 1, 0],
 [ 4,33, 1],
 [ 0, 0,15]]

---

👥 3. Student Clustering

The project uses K-Means clustering to analyze groups of students with similar characteristics.

The Elbow Method was evaluated for different values of K.

K| Inertia
2| 21636.91
3| 20791.69
4| 20001.60
5| 19284.03
6| 18928.12
7| 18451.46
8| 18055.65

The project uses 4 clusters for student profiling.

Cluster Sizes

Cluster| Students
0| 20
1| 64
2| 166
3| 145

Cluster Profile

Cluster| Age| Study Time| Failures| Absences
0| 17.85| 1.40| 1.30| 7.65
1| 17.72| 1.84| 0.48| 3.92
2| 16.45| 2.34| 0.31| 6.25
3| 16.37| 1.86| 0.16| 5.61

---

⚠️ 4. Early-Warning Analysis

The project also contains an early-warning analysis.

An early-warning Random Forest experiment was evaluated without using "G1" and "G2".

Result

Metric| Result
MAE| 3.0016
RMSE| 3.7671
R² Score| 0.3079

This experiment evaluates prediction when earlier grade information is not used.

---

🧠 5. SHAP Explainability

The project includes SHAP-based explanation to help understand the model's predictions.

The Streamlit application can display an explanation for an individual student's prediction.

Example application output included:

- Predicted final grade
- Support level
- SHAP explanation
- Personalized recommendations

This makes the prediction easier to interpret instead of showing only a numerical result.

---

💡 6. Personalized Recommendations

The project contains a recommendation component that generates personalized academic recommendations based on the student's analysis.

The recommendations are displayed through the Streamlit application along with the prediction and support level.

---

🌐 7. Streamlit Application

The project contains the main application:

app/app.py

The Streamlit application connects the trained models, preprocessing components, SHAP explanation, prediction and recommendation functionality.

Run the application

streamlit run app/app.py

The application opens in the browser and allows the user to interact with the Student Performance AI system.

---

🏗️ Project Structure

Student_Performance_AI/
│
├── app/
│   └── app.py
│
├── data/
│   └── student-mat.csv
│
├── models/
│   ├── tuned_random_forest_regressor.pkl
│   ├── svm_classifier.pkl
│   └── preprocessing/model files
│
├── notebooks/
│
├── reports/
│
├── src/
│   ├── data analysis
│   ├── model training
│   ├── clustering
│   ├── early-warning analysis
│   ├── recommendations
│   └── SHAP analysis
│
├── tests/
│   └── test_recommendations.py
│
├── requirements.txt
├── README.md
└── PROGRAM_BOOK.md

---

🛠️ Technologies Used

Programming Language

- Python

Data Processing

- Pandas
- NumPy

Machine Learning

- Scikit-learn

Visualization

- Matplotlib
- Seaborn

Explainable AI

- SHAP

Web Application

- Streamlit

Model Saving

- Joblib

Development

- Visual Studio Code
- Git
- GitHub

---

🔬 Machine Learning Workflow

Student Dataset
       │
       ▼
Data Preprocessing
       │
       ▼
Feature Processing
       │
       ▼
Train / Test Split
       │
       ├───────────────┬────────────────┐
       ▼               ▼                ▼
  Regression     Classification     Clustering
       │               │                │
       ▼               ▼                ▼
  G3 Prediction   Support Level    Student Groups
       │               │                │
       └───────────────┼────────────────┘
                       ▼
                Model Analysis
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
           SHAP          Recommendations
             │                   │
             └─────────┬─────────┘
                       ▼
                Streamlit App

---

📈 Regression Model Summary

The project tested multiple regression results before selecting the tuned model.

MAE| RMSE| R²
1.6467| 2.3784| 0.7241
1.3412| 2.4036| 0.7182
1.1779| 2.0014| 0.8047
1.1477| 2.0147| 0.8020
1.2051| 2.0249| 0.8000
1.1770| 1.9978| 0.8054

---

📋 Classification Model Evaluation

The classification models evaluated in the project were:

Model
Logistic Regression
Decision Tree
Random Forest
Gradient Boosting
SVM

The SVM model achieved the highest recorded performance:

Accuracy: 92.41%

---

⚙️ Setup

Clone the repository

git clone YOUR_GITHUB_REPOSITORY_URL

cd Student_Performance_AI

Install dependencies

pip install -r requirements.txt

Run the application

streamlit run app/app.py

---

🧪 Testing

The project contains:

tests/test_recommendations.py

Testing can be performed using:

pytest tests

---

📚 Documentation

Detailed project documentation is available in:

PROGRAM_BOOK.md

The repository also contains the source code, trained model files, dataset, reports, notebooks and tests.

---

🎓 Project Summary

Student Performance AI combines:

Regression
    +
Classification
    +
Clustering
    +
Early-Warning Analysis
    +
SHAP Explainability
    +
Personalized Recommendations
    +
Streamlit

to create an integrated Machine Learning system for student academic analysis.

---

👨‍💻 Developer

Venky

Roll No: 232G1A0489

Student Performance AI