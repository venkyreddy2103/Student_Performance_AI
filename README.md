🎓 Student Performance AI

Machine Learning Project for Student Performance Prediction, Support-Level Classification & Student Clustering

"Python" (https://img.shields.io/badge/Python-3.14-blue?logo=python&logoColor=white)
"Machine Learning" (https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange?logo=scikit-learn&logoColor=white)
"Pandas" (https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
"NumPy" (https://img.shields.io/badge/NumPy-Data%20Processing-013243?logo=numpy&logoColor=white)
"Streamlit" (https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit&logoColor=white)
"GitHub" (https://img.shields.io/badge/GitHub-Repository-181717?logo=github&logoColor=white)

---

👨‍💻 Project Information

Student Name: Venky
Roll Number: 232G1A0489
Branch: Electronics and Communication Engineering (ECE)
Project: Student Performance AI
Type: Machine Learning Project

---

📌 About The Project

Student Performance AI is a Machine Learning project designed to analyze student academic performance and provide meaningful predictions and classifications.

The project uses student academic and personal information to perform three major Machine Learning tasks:

- 📈 Student Performance Prediction – predicts the final student grade.
- 🎯 Support-Level Classification – identifies students who may require High, Medium, or Low academic support.
- 👥 Student Clustering – groups students with similar characteristics using K-Means clustering.

The project also includes a Streamlit web application that provides an interactive interface for using the trained Machine Learning models.

---

🎯 Project Objectives

The main objectives of this project are:

1. Analyze student academic performance.
2. Predict the student's final grade.
3. Classify students according to their required support level.
4. Identify groups of students with similar characteristics.
5. Compare different Machine Learning algorithms.
6. Build an interactive web application using Streamlit.
7. Provide useful insights from student data.

---

📊 Dataset

The project uses the Student Performance dataset stored as:

data/student-mat.csv

The dataset contains 395 student records and 32 original features.

The target variable for regression is:

G3

"G3" represents the student's final grade.

---

🤖 Machine Learning Tasks

1️⃣ Student Performance Prediction

This task treats the final grade G3 as a regression target.

Several Machine Learning models were evaluated and optimized.

🔧 Best Model

The tuned Random Forest model achieved:

Metric| Result
MAE| 1.1770
RMSE| 1.9978
R² Score| 0.8054

The model achieved approximately 80.5% R², indicating a strong relationship between the selected features and final student performance.

Best Hyperparameters

n_estimators = 300
min_samples_split = 2
min_samples_leaf = 1
max_features = 1.0
max_depth = 10

---

2️⃣ 🎯 Support-Level Classification

Students are classified into three support categories based on their final grade:

Final Grade| Support Level
G3 < 10| 🔴 High Support
10 ≤ G3 < 15| 🟡 Medium Support
G3 ≥ 15| 🟢 Low Support

📊 Dataset Distribution

Support Level| Students
🟡 Medium Support| 192
🔴 High Support| 130
🟢 Low Support| 73

🏆 Best Classification Model

The Support Vector Machine (SVM) achieved the best test performance:

Metric| Result
Accuracy| 92.41%
Precision| 92.86%
F1 Score| 92.39%

Confusion Matrix

[[25, 1, 0],
 [ 4,33, 1],
 [ 0, 0,15]]

---

3️⃣ 👥 Student Clustering

K-Means clustering was used to identify groups of students with similar characteristics.

Different values of K were evaluated using the Elbow Method.

The selected clustering configuration used:

Number of Clusters = 4

Cluster Sizes

Cluster| Students
Cluster 0| 20
Cluster 1| 64
Cluster 2| 166
Cluster 3| 145

📋 Cluster Profiles

Important characteristics analyzed include:

- Age
- Study time
- Previous failures
- Absences

Cluster| Avg. Age| Study Time| Failures| Absences
Cluster 0| 17.85| 1.40| 1.30| 7.65
Cluster 1| 17.72| 1.84| 0.48| 3.92
Cluster 2| 16.45| 2.34| 0.31| 6.25
Cluster 3| 16.37| 1.86| 0.16| 5.61

---

🧠 Data Preprocessing

The project uses a preprocessing pipeline to prepare the dataset for Machine Learning.

Preprocessing includes:

- 🔹 Separating features and target
- 🔹 Train-test splitting
- 🔹 Handling categorical features
- 🔹 One-Hot Encoding
- 🔹 Handling unknown categorical values
- 🔹 Passing numerical features through
- 🔹 Model training and evaluation

The preprocessing pipeline converted the original 32 features into 58 processed features after encoding.

---

📁 Project Structure

Student_Performance_AI/
│
├── 📂 data/
│   └── student-mat.csv
│
├── 📂 notebooks/
│   └── Machine Learning notebooks
│
├── 📂 src/
│   ├── check_dataset.py
│   └── Machine Learning source files
│
├── 📂 models/
│   └── Trained Machine Learning models
│
├── 📂 app/
│   └── app.py
│
├── 📂 tests/
│   └── Project tests
│
├── 📂 reports/
│   └── Analysis and project reports
│
├── 📄 requirements.txt
└── 📄 README.md

---

🛠️ Technologies Used

Technology| Purpose
🐍 Python| Programming language
🐼 Pandas| Data analysis
🔢 NumPy| Numerical operations
🤖 Scikit-learn| Machine Learning
🌐 Streamlit| Web application
💾 Joblib| Model saving/loading
📊 Matplotlib| Data visualization
🗃️ Git| Version control
🐙 GitHub| Project hosting

---

🚀 How To Run The Project

1. Clone the repository

git clone YOUR_GITHUB_REPOSITORY_LINK

2. Open the project

cd Student_Performance_AI

3. Create a virtual environment

python -m venv .venv

Windows

.venv\Scripts\activate

4. Install dependencies

pip install -r requirements.txt

5. Run the Streamlit application

streamlit run app/app.py

The Streamlit application will open in your browser.

---

🖥️ Streamlit Application

The project includes an interactive Streamlit application for working with the trained Machine Learning models.

The application can be used to:

- 📈 Predict student performance
- 🎯 Identify support level
- 📊 View model-related results
- 📉 Visualize project insights
- 🤖 Interact with the Machine Learning system

---

📈 Model Performance Summary

Regression

Model| MAE| RMSE| R²
Initial Model| 1.6467| 2.3784| 0.7241
Model 2| 1.3412| 2.4036| 0.7182
Model 3| 1.1779| 2.0014| 0.8047
Model 4| 1.1477| 2.0147| 0.8020
Model 5| 1.2051| 2.0249| 0.8000
🏆 Tuned Random Forest| 1.1770| 1.9978| 0.8054

---

🎯 Classification

Model| Accuracy| F1 Score
Logistic Regression| 91.14%| 91.11%
Decision Tree| 89.87%| —
Random Forest| 87.34%| —
Gradient Boosting| 89.87%| —
🏆 SVM| 92.41%| 92.39%

---

📚 Key Learning Outcomes

Through this project, the following Machine Learning concepts were implemented:

- ✅ Data preprocessing
- ✅ Exploratory data analysis
- ✅ Feature encoding
- ✅ Train-test splitting
- ✅ Regression
- ✅ Classification
- ✅ Hyperparameter tuning
- ✅ RandomizedSearchCV
- ✅ Model evaluation
- ✅ Confusion matrix
- ✅ F1 Score
- ✅ K-Means clustering
- ✅ Elbow Method
- ✅ Data visualization
- ✅ Model serialization using Joblib
- ✅ Streamlit application development
- ✅ Git and GitHub version control

---

🔮 Future Improvements

Possible future improvements include:

- 🔹 Improve prediction accuracy
- 🔹 Add more advanced Machine Learning models
- 🔹 Add additional visualizations
- 🔹 Improve the Streamlit UI
- 🔹 Add real-time student performance monitoring
- 🔹 Deploy the application online
- 🔹 Add authentication and user management
- 🔹 Add more educational datasets

---

👨‍🎓 Author

Venky
B.Tech – Electronics and Communication Engineering
Roll No: 232G1A0489

---

🔗 Useful Links

- 🐙 GitHub: https://github.com/venkyreddy2103/Student_Performance_AI.git
- 🌐 Live Streamlit App: https://studentperformanceai-venkatareddy.streamlit.app/
- 📚 Project Documentation: This README

---

⭐ Project Status

🚧 Active Machine Learning Project

Built as an academic Machine Learning project to analyze and predict student performance.

---

🙌 Acknowledgements

This project was developed using Python and open-source Machine Learning libraries including Pandas, NumPy, Scikit-learn and Streamlit.

---

⭐ If you find this project useful, consider giving the repository a star!