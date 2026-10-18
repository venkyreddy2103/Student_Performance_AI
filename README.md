# 🎓 Intelligent Student Performance AI (SP-AI)

An end-to-end Machine Learning, Explainable AI (SHAP), and personalized academic intervention system built on the UCI Student Performance dataset.

> 📖 **Comprehensive Documentation:** For the full technical dossier, mathematical formulations, benchmarks, architecture diagrams, and operational runbook, please consult the [Official Program Book](file:///c:/Users/Venkata%20Reddy/OneDrive/Documents/Student_Performence_AI/PROGRAM_BOOK.md).

---

## 🚀 Key Highlights

- **Dual-Model Predictive Core:**
  - **Regression:** Tuned Random Forest Regressor predicts continuous final grade ($G3 \in [0, 20]$) with **$R^2 = 0.8179$** and **$\text{RMSE} = 1.9326$**.
  - **Classification:** Support Vector Machine (RBF Kernel) classifies students into **High Support**, **Medium Support**, and **Low Support** tiers with **$93.67\%$ Accuracy** and **$0.9908$ Multiclass ROC-AUC**.
- **Explainable AI (XAI):** Tree-based and Kernel SHAP attributions reveal positive and negative feature drivers for every prediction.
- **Pedagogical Recommendation Engine:** Rule-based decision logic generating actionable, personalized study plans and institutional support guidance.
- **Interactive Web App:** Full-featured 5-tab Streamlit dashboard with real-time inference, file upload validator, and report export.
- **Automated Test Suite:** Comprehensive pytest suite covering feature engineering, model inference bounds, recommendations, and robustness.

---

## 📁 Repository Overview

```
Student_Performence_AI/
├── PROGRAM_BOOK.md                # 📖 Official Comprehensive Program Book
├── requirements.txt               # Project dependencies
├── data/
│   └── student-mat.csv            # UCI Student Performance dataset (395 records, 33 attributes)
├── models/                        # Serialized production models & preprocessors (.pkl)
├── src/                           # ML pipeline, feature engineering, clustering, SHAP, and recommendations
├── app/
│   └── app.py                     # 5-tab Streamlit web application
├── reports/                       # Visualizations, SHAP importances, and robustness test results
├── notebooks/
│   └── eda.py                     # Exploratory Data Analysis routines
└── tests/                         # Pytest unit and regression tests
```

---

## ⚡ Quick Start

### 1. Activate Environment & Install Dependencies
```powershell
# Activate existing virtual environment
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 2. Run Master Training Pipeline
```powershell
python src/feature_engineering.py
```

### 3. Run Quality Assurance Tests
```powershell
pytest -v
```

### 4. Launch the Interactive Web Dashboard
```powershell
streamlit run app/app.py
```
Open your browser at `http://localhost:8501`.

---

## 📜 License & Citation
- **Dataset:** UCI Machine Learning Repository (Cortez & Silva, 2008).
- **Project Documentation:** See [PROGRAM_BOOK.md](file:///c:/Users/Venkata%20Reddy/OneDrive/Documents/Student_Performence_AI/PROGRAM_BOOK.md).
