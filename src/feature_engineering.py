# ============================================================
# STUDENT PERFORMANCE AI
# Complete Feature Engineering + ML Pipeline
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------- SCIKIT-LEARN IMPORTS --------------------

from sklearn.model_selection import train_test_split, RandomizedSearchCV

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.tree import (
    DecisionTreeRegressor,
    DecisionTreeClassifier
)

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.svm import SVR, SVC

from sklearn.cluster import KMeans

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score
)


# ============================================================
# 1. FEATURE ENGINEERING
# ============================================================

def create_features(df):
    """
    Create meaningful derived features for the student dataset.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Assessment trend
    # Difference between second and first assessment
    # --------------------------------------------------------

    df["assessment_trend"] = df["G2"] - df["G1"]

    # --------------------------------------------------------
    # Previous assessment average
    # --------------------------------------------------------

    df["previous_grade_avg"] = (
        df["G1"] + df["G2"]
    ) / 2

    # --------------------------------------------------------
    # Attendance-related feature
    # --------------------------------------------------------

    df["absence_band"] = pd.cut(
        df["absences"],
        bins=[-1, 5, 10, 20, float("inf")],
        labels=[
            "Low",
            "Moderate",
            "High",
            "Very High"
        ]
    ).astype(str)

    # --------------------------------------------------------
    # Academic difficulty indicator
    # --------------------------------------------------------

    df["failure_risk"] = (
        df["failures"] > 0
    ).astype(int)

    # --------------------------------------------------------
    # Engagement consistency
    # --------------------------------------------------------

    df["engagement_score"] = (
        df["studytime"]
        + df["famrel"]
        + df["freetime"]
        + df["goout"]
    ) / 4

    return df


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv(
    "data/student-mat.csv",
    sep=";"
)

print("\n================================================")
print("DATASET LOADED")
print("================================================")

print("Dataset shape:", df.shape)


# ============================================================
# 3. CREATE ENGINEERED FEATURES
# ============================================================

df = create_features(df)

print("\n================================================")
print("FEATURE ENGINEERING")
print("================================================")

print("Engineered features added:")
print("- assessment_trend")
print("- previous_grade_avg")
print("- absence_band")
print("- failure_risk")
print("- engagement_score")


# ============================================================
# 4. REGRESSION TARGET AND FEATURES
# ============================================================

y = df["G3"]

X = df.drop(
    columns=["G3"]
)

print("\n===== FEATURE ENGINEERING =====")

print("Feature shape:", X.shape)
print("Target shape:", y.shape)

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print(y.name)


# ============================================================
# 5. REGRESSION TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n===== TRAIN TEST SPLIT =====")

print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)

print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)


# ============================================================
# 6. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numerical_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "string"]
).columns.tolist()

print("\n===== FEATURE TYPES =====")

print("\nNumerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)

print(
    "\nNumber of numerical features:",
    len(numerical_features)
)

print(
    "Number of categorical features:",
    len(categorical_features)
)


# ============================================================
# 7. REGRESSION PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ],
    remainder="passthrough"
)

# IMPORTANT:
# Fit only on training data

X_train_processed = preprocessor.fit_transform(
    X_train
)

X_test_processed = preprocessor.transform(
    X_test
)

print("\n===== PREPROCESSING =====")

print(
    "Original training shape:",
    X_train.shape
)

print(
    "Processed training shape:",
    X_train_processed.shape
)

print(
    "Original testing shape:",
    X_test.shape
)

print(
    "Processed testing shape:",
    X_test_processed.shape
)


# ============================================================
# 8. REGRESSION BASELINE
# ============================================================

# Predict the mean training grade for every test student

baseline_prediction = y_train.mean()

baseline_pred = np.full(
    len(y_test),
    baseline_prediction
)

baseline_mae = mean_absolute_error(
    y_test,
    baseline_pred
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        baseline_pred
    )
)

baseline_r2 = r2_score(
    y_test,
    baseline_pred
)

print("\n===== REGRESSION BASELINE =====")

print(
    "Baseline prediction:",
    round(baseline_prediction, 2)
)

print(
    "MAE:",
    round(baseline_mae, 4)
)

print(
    "RMSE:",
    round(baseline_rmse, 4)
)

print(
    "R2 Score:",
    round(baseline_r2, 4)
)


# ============================================================
# 9. LINEAR REGRESSION
# ============================================================

linear_model = LinearRegression()

linear_model.fit(
    X_train_processed,
    y_train
)

linear_pred = linear_model.predict(
    X_test_processed
)

mae = mean_absolute_error(
    y_test,
    linear_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        linear_pred
    )
)

r2 = r2_score(
    y_test,
    linear_pred
)

print("\n===== LINEAR REGRESSION RESULTS =====")

print("MAE:", mae)
print("RMSE:", rmse)
print("R² Score:", r2)


# ============================================================
# 10. DECISION TREE REGRESSION
# ============================================================

tree_model = DecisionTreeRegressor(
    random_state=42,
    max_depth=5
)

tree_model.fit(
    X_train_processed,
    y_train
)

tree_pred = tree_model.predict(
    X_test_processed
)

tree_mae = mean_absolute_error(
    y_test,
    tree_pred
)

tree_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        tree_pred
    )
)

tree_r2 = r2_score(
    y_test,
    tree_pred
)

print("\n===== DECISION TREE RESULTS =====")

print("MAE:", tree_mae)
print("RMSE:", tree_rmse)
print("R² Score:", tree_r2)


# ============================================================
# 11. RANDOM FOREST REGRESSION
# ============================================================

rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

rf_model.fit(
    X_train_processed,
    y_train
)

rf_pred = rf_model.predict(
    X_test_processed
)

rf_mae = mean_absolute_error(
    y_test,
    rf_pred
)

rf_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        rf_pred
    )
)

rf_r2 = r2_score(
    y_test,
    rf_pred
)

print("\n===== RANDOM FOREST RESULTS =====")

print("MAE:", rf_mae)
print("RMSE:", rf_rmse)
print("R² Score:", rf_r2)


# ============================================================
# 12. GRADIENT BOOSTING REGRESSION
# ============================================================

gb_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

gb_model.fit(
    X_train_processed,
    y_train
)

gb_pred = gb_model.predict(
    X_test_processed
)

gb_mae = mean_absolute_error(
    y_test,
    gb_pred
)

gb_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        gb_pred
    )
)

gb_r2 = r2_score(
    y_test,
    gb_pred
)

print("\n===== GRADIENT BOOSTING RESULTS =====")

print("MAE:", gb_mae)
print("RMSE:", gb_rmse)
print("R² Score:", gb_r2)


# ============================================================
# 13. SVR REGRESSION
# ============================================================

svr_model = SVR(
    kernel="rbf",
    C=10,
    epsilon=0.1
)

svr_model.fit(
    X_train_processed,
    y_train
)

svr_pred = svr_model.predict(
    X_test_processed
)

svr_mae = mean_absolute_error(
    y_test,
    svr_pred
)

svr_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        svr_pred
    )
)

svr_r2 = r2_score(
    y_test,
    svr_pred
)

print("\n===== SVR RESULTS =====")

print("MAE:", svr_mae)
print("RMSE:", svr_rmse)
print("R² Score:", svr_r2)


# ============================================================
# 14. RANDOM FOREST HYPERPARAMETER TUNING
# ============================================================

rf_base = RandomForestRegressor(
    random_state=42
)

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [
        None,
        5,
        10,
        15,
        20
    ],
    "min_samples_split": [
        2,
        5,
        10
    ],
    "min_samples_leaf": [
        1,
        2,
        4
    ],
    "max_features": [
        0.5,
        0.75,
        1.0
    ]
}

rf_search = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=param_grid,
    n_iter=15,
    scoring="neg_mean_absolute_error",
    cv=5,
    random_state=42,
    n_jobs=-1
)

rf_search.fit(
    X_train_processed,
    y_train
)

print("\n===== RANDOM FOREST TUNING =====")

print("Best parameters:")
print(rf_search.best_params_)

print("\nBest CV MAE:")
print(-rf_search.best_score_)


# ============================================================
# 15. TUNED RANDOM FOREST TEST RESULTS
# ============================================================

best_rf = rf_search.best_estimator_

best_rf_pred = best_rf.predict(
    X_test_processed
)

best_rf_mae = mean_absolute_error(
    y_test,
    best_rf_pred
)

best_rf_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        best_rf_pred
    )
)

best_rf_r2 = r2_score(
    y_test,
    best_rf_pred
)

print("\n===== TUNED RANDOM FOREST TEST RESULTS =====")

print("MAE:", best_rf_mae)
print("RMSE:", best_rf_rmse)
print("R² Score:", best_rf_r2)


# ============================================================
# 16. SUPPORT LEVEL CREATION
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

print("\n===== SUPPORT GROUP DISTRIBUTION =====")

print(
    df["Support_Level"].value_counts()
)


# ============================================================
# 17. CLASSIFICATION DATA
# ============================================================

X_class = df.drop(
    columns=[
        "G3",
        "Support_Level"
    ]
)

y_class = df["Support_Level"]

print("\n===== CLASSIFICATION DATA =====")

print(
    "Features shape:",
    X_class.shape
)

print(
    "Target shape:",
    y_class.shape
)

print("\nTarget classes:")

print(
    y_class.value_counts()
)


# ============================================================
# 18. CLASSIFICATION TRAIN / TEST SPLIT
# ============================================================

X_class_train, X_class_test, y_class_train, y_class_test = train_test_split(
    X_class,
    y_class,
    test_size=0.20,
    random_state=42,
    stratify=y_class
)

print("\n===== CLASSIFICATION TRAIN TEST SPLIT =====")

print(
    "Training features:",
    X_class_train.shape
)

print(
    "Testing features:",
    X_class_test.shape
)

print(
    "Training target:",
    y_class_train.shape
)

print(
    "Testing target:",
    y_class_test.shape
)

print("\nTraining class distribution:")

print(
    y_class_train.value_counts()
)

print("\nTesting class distribution:")

print(
    y_class_test.value_counts()
)


# ============================================================
# 19. CLASSIFICATION BASELINE
# ============================================================

# IMPORTANT:
# This comes AFTER y_class_train and y_class_test
# have been created.

majority_class = y_class_train.mode()[0]

baseline_class_prediction = [
    majority_class
] * len(y_class_test)

baseline_accuracy = accuracy_score(
    y_class_test,
    baseline_class_prediction
)

baseline_precision = precision_score(
    y_class_test,
    baseline_class_prediction,
    average="weighted",
    zero_division=0
)

baseline_recall = recall_score(
    y_class_test,
    baseline_class_prediction,
    average="weighted",
    zero_division=0
)

baseline_f1 = f1_score(
    y_class_test,
    baseline_class_prediction,
    average="weighted",
    zero_division=0
)

print("\n===== CLASSIFICATION BASELINE =====")

print(
    "Majority class:",
    majority_class
)

print(
    "Accuracy:",
    round(baseline_accuracy, 4)
)

print(
    "Precision:",
    round(baseline_precision, 4)
)

print(
    "Recall:",
    round(baseline_recall, 4)
)

print(
    "F1 Score:",
    round(baseline_f1, 4)
)


# ============================================================
# 20. CLASSIFICATION PREPROCESSING
# ============================================================

class_numerical_features = X_class_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

class_categorical_features = X_class_train.select_dtypes(
    include=["object", "string"]
).columns.tolist()

class_preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            class_categorical_features
        )
    ],
    remainder="passthrough"
)

X_class_train_processed = class_preprocessor.fit_transform(
    X_class_train
)

X_class_test_processed = class_preprocessor.transform(
    X_class_test
)

print("\n===== CLASSIFICATION PREPROCESSING =====")

print(
    "Original training shape:",
    X_class_train.shape
)

print(
    "Processed training shape:",
    X_class_train_processed.shape
)

print(
    "Original testing shape:",
    X_class_test.shape
)

print(
    "Processed testing shape:",
    X_class_test_processed.shape
)


# ============================================================
# 21. LOGISTIC REGRESSION CLASSIFIER
# ============================================================

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_model.fit(
    X_class_train_processed,
    y_class_train
)

logistic_pred = logistic_model.predict(
    X_class_test_processed
)

logistic_accuracy = accuracy_score(
    y_class_test,
    logistic_pred
)

logistic_precision = precision_score(
    y_class_test,
    logistic_pred,
    average="weighted",
    zero_division=0
)

logistic_recall = recall_score(
    y_class_test,
    logistic_pred,
    average="weighted",
    zero_division=0
)

logistic_f1 = f1_score(
    y_class_test,
    logistic_pred,
    average="weighted",
    zero_division=0
)

print("\n===== LOGISTIC REGRESSION RESULTS =====")

print("Accuracy:", logistic_accuracy)
print("Precision:", logistic_precision)
print("Recall:", logistic_recall)
print("F1 Score:", logistic_f1)


# ============================================================
# 22. LOGISTIC CONFUSION MATRIX
# ============================================================

class_labels = [
    "High Support",
    "Medium Support",
    "Low Support"
]

logistic_cm = confusion_matrix(
    y_class_test,
    logistic_pred,
    labels=class_labels
)

print("\n===== LOGISTIC CONFUSION MATRIX =====")

print(logistic_cm)


# ============================================================
# 23. DECISION TREE CLASSIFIER
# ============================================================

decision_tree_model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

decision_tree_model.fit(
    X_class_train_processed,
    y_class_train
)

decision_tree_pred = decision_tree_model.predict(
    X_class_test_processed
)

dt_accuracy = accuracy_score(
    y_class_test,
    decision_tree_pred
)

dt_precision = precision_score(
    y_class_test,
    decision_tree_pred,
    average="weighted",
    zero_division=0
)

dt_recall = recall_score(
    y_class_test,
    decision_tree_pred,
    average="weighted",
    zero_division=0
)

dt_f1 = f1_score(
    y_class_test,
    decision_tree_pred,
    average="weighted",
    zero_division=0
)

print("\n===== DECISION TREE CLASSIFIER RESULTS =====")

print("Accuracy:", dt_accuracy)
print("Precision:", dt_precision)
print("Recall:", dt_recall)
print("F1 Score:", dt_f1)


# ============================================================
# 24. RANDOM FOREST CLASSIFIER
# ============================================================

random_forest_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

random_forest_model.fit(
    X_class_train_processed,
    y_class_train
)

random_forest_pred = random_forest_model.predict(
    X_class_test_processed
)

rf_accuracy = accuracy_score(
    y_class_test,
    random_forest_pred
)

rf_precision = precision_score(
    y_class_test,
    random_forest_pred,
    average="weighted",
    zero_division=0
)

rf_recall = recall_score(
    y_class_test,
    random_forest_pred,
    average="weighted",
    zero_division=0
)

rf_f1 = f1_score(
    y_class_test,
    random_forest_pred,
    average="weighted",
    zero_division=0
)

print("\n===== RANDOM FOREST CLASSIFIER RESULTS =====")

print("Accuracy:", rf_accuracy)
print("Precision:", rf_precision)
print("Recall:", rf_recall)
print("F1 Score:", rf_f1)


# ============================================================
# 25. GRADIENT BOOSTING CLASSIFIER
# ============================================================

gradient_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

gradient_model.fit(
    X_class_train_processed,
    y_class_train
)

gradient_pred = gradient_model.predict(
    X_class_test_processed
)

gb_accuracy = accuracy_score(
    y_class_test,
    gradient_pred
)

gb_precision = precision_score(
    y_class_test,
    gradient_pred,
    average="weighted",
    zero_division=0
)

gb_recall = recall_score(
    y_class_test,
    gradient_pred,
    average="weighted",
    zero_division=0
)
gb_f1 = f1_score(
    y_class_test,
    gradient_pred,
    average="weighted",
    zero_division=0
)

print("\n===== GRADIENT BOOSTING CLASSIFIER RESULTS =====")

print("Accuracy:", gb_accuracy)
print("Precision:", gb_precision)
print("Recall:", gb_recall)
print("F1 Score:", gb_f1)


# ============================================================
# 26. SVM CLASSIFIER
# ============================================================

svm_model = SVC(
    kernel="rbf",
    C=10,
    probability=True,
    random_state=42
)

svm_model.fit(
    X_class_train_processed,
    y_class_train
)

svm_pred = svm_model.predict(
    X_class_test_processed
)

svm_accuracy = accuracy_score(
    y_class_test,
    svm_pred
)

svm_precision = precision_score(
    y_class_test,
    svm_pred,
    average="weighted",
    zero_division=0
)

svm_recall = recall_score(
    y_class_test,
    svm_pred,
    average="weighted",
    zero_division=0
)

svm_f1 = f1_score(
    y_class_test,
    svm_pred,
    average="weighted",
    zero_division=0
)

print("\n===== SVM CLASSIFIER RESULTS =====")

print("Accuracy:", svm_accuracy)
print("Precision:", svm_precision)
print("Recall:", svm_recall)
print("F1 Score:", svm_f1)


# ============================================================
# 27. SVM CONFUSION MATRIX
# ============================================================

svm_cm = confusion_matrix(
    y_class_test,
    svm_pred,
    labels=class_labels
)

print("\n===== SVM CONFUSION MATRIX =====")

print(svm_cm)


# ============================================================
# 28. SVM ROC-AUC
# ============================================================

svm_probabilities = svm_model.predict_proba(
    X_class_test_processed
)

svm_roc_auc = roc_auc_score(
    y_class_test,
    svm_probabilities,
    multi_class="ovr",
    average="weighted",
    labels=svm_model.classes_
)

print("\n===== SVM ROC-AUC =====")

print(
    "Weighted ROC-AUC:",
    svm_roc_auc
)


# ============================================================
# 29. CLASSIFICATION MODEL COMPARISON
# ============================================================

classification_comparison = pd.DataFrame({

    "Model": [
        "Majority Baseline",
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
        "SVM"
    ],

    "Accuracy": [
        baseline_accuracy,
        logistic_accuracy,
        dt_accuracy,
        rf_accuracy,
        gb_accuracy,
        svm_accuracy
    ],

    "Precision": [
        baseline_precision,
        logistic_precision,
        dt_precision,
        rf_precision,
        gb_precision,
        svm_precision
    ],

    "Recall": [
        baseline_recall,
        logistic_recall,
        dt_recall,
        rf_recall,
        gb_recall,
        svm_recall
    ],

    "F1 Score": [
        baseline_f1,
        logistic_f1,
        dt_f1,
        rf_f1,
        gb_f1,
        svm_f1
    ]
})

print("\n===== CLASSIFICATION MODEL COMPARISON =====")

print(
    classification_comparison
    .round(3)
    .to_string(index=False)
)


# ============================================================
# 30. BEST CLASSIFICATION MODEL
# ============================================================

best_classification_model = (
    classification_comparison
    .loc[
        classification_comparison["F1 Score"].idxmax(),
        "Model"
    ]
)

print("\n===== BEST CLASSIFICATION MODEL =====")

print(
    "Best by F1 Score:",
    best_classification_model
)


# ============================================================
# 31. CLUSTERING DATA
# ============================================================

X_cluster = df.drop(
    columns=[
        "G3",
        "Support_Level"
    ]
)

print("\n===== CLUSTERING DATA =====")

print(
    "Clustering features shape:",
    X_cluster.shape
)

print("\nFeatures used for clustering:")

print(
    X_cluster.columns.tolist()
)


# ============================================================
# 32. CLUSTERING PREPROCESSING
# ============================================================

cluster_numerical_features = X_cluster.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

cluster_categorical_features = X_cluster.select_dtypes(
    include=["object", "string"]
).columns.tolist()

cluster_preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            cluster_categorical_features
        )
    ],
    remainder="passthrough"
)

X_cluster_encoded = cluster_preprocessor.fit_transform(
    X_cluster
)

scaler = StandardScaler(
    with_mean=False
)

X_cluster_scaled = scaler.fit_transform(
    X_cluster_encoded
)

print("\n===== CLUSTERING PREPROCESSING =====")

print(
    "Original shape:",
    X_cluster.shape
)

print(
    "Encoded shape:",
    X_cluster_encoded.shape
)

print(
    "Scaled shape:",
    X_cluster_scaled.shape
)


# ============================================================
# 33. ELBOW METHOD
# ============================================================

inertia_values = []

for k in range(2, 9):

    kmeans_test = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans_test.fit(
        X_cluster_scaled
    )

    inertia_values.append(
        kmeans_test.inertia_
    )


print("\n===== ELBOW METHOD =====")

for k, inertia in zip(
    range(2, 9),
    inertia_values
):

    print(
        "K =",
        k,
        "Inertia =",
        inertia
    )


# Plot elbow

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    range(2, 9),
    inertia_values,
    marker="o"
)

plt.xlabel(
    "Number of Clusters (K)"
)

plt.ylabel(
    "Inertia"
)

plt.title(
    "Elbow Method for K-Means Clustering"
)

plt.grid(True)

plt.show()


# ============================================================
# 34. K-MEANS CLUSTERING
# ============================================================

# Keep K=4 as the project clustering configuration

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

cluster_labels = kmeans.fit_predict(
    X_cluster_scaled
)

print("\n===== K-MEANS RESULTS =====")

print(
    "Number of clusters:",
    4
)

print("\nStudents in each cluster:")

print(
    pd.Series(
        cluster_labels
    )
    .value_counts()
    .sort_index()
)


# ============================================================
# 35. CLUSTER PROFILING
# ============================================================

cluster_profile = X_cluster.copy()

cluster_profile["Cluster"] = cluster_labels

profile = (
    cluster_profile
    .groupby("Cluster")[
        [
            "age",
            "studytime",
            "failures",
            "absences"
        ]
    ]
    .mean()
)

print("\n===== CLUSTER PROFILE =====")

print(
    profile.round(2).to_string()
)

print("\n===== CLUSTER SIZES =====")

print(
    cluster_profile["Cluster"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 36. CLUSTER VISUALIZATION
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.scatter(
    cluster_profile["studytime"],
    cluster_profile["failures"],
    c=cluster_profile["Cluster"],
    cmap="viridis"
)

plt.xlabel(
    "Study Time"
)

plt.ylabel(
    "Previous Failures"
)

plt.title(
    "Student Clusters: Study Time vs Previous Failures"
)

plt.colorbar(
    label="Cluster"
)

plt.show()


# ============================================================
# 37. SAVE MODELS
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)


# Regression model

joblib.dump(
    best_rf,
    "models/tuned_random_forest_regressor.pkl"
)


# Regression preprocessor

joblib.dump(
    preprocessor,
    "models/regression_preprocessor.pkl"
)


# Classification model

joblib.dump(
    svm_model,
    "models/svm_classifier.pkl"
)


# Classification preprocessor

joblib.dump(
    class_preprocessor,
    "models/classification_preprocessor.pkl"
)


print("\n===== MODEL SAVING =====")

print(
    "Tuned Random Forest saved successfully."
)

print(
    "Regression preprocessor saved successfully."
)

print(
    "SVM classifier saved successfully."
)

print(
    "Classification preprocessor saved successfully."
)


# ============================================================
# 38. REGRESSION RESIDUAL ANALYSIS
# ============================================================

residuals = (
    y_test - best_rf_pred
)

print("\n===== RESIDUAL ANALYSIS =====")

print(
    "Mean residual:",
    residuals.mean()
)

print(
    "Minimum residual:",
    residuals.min()
)

print(
    "Maximum residual:",
    residuals.max()
)


# Residual plot

plt.figure(
    figsize=(8, 5)
)

plt.scatter(
    best_rf_pred,
    residuals
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel(
    "Predicted Final Score"
)

plt.ylabel(
    "Residual (Actual - Predicted)"
)

plt.title(
    "Random Forest Residual Analysis"
)

plt.grid(True)

plt.show()


# ============================================================
# 39. REGRESSION MODEL COMPARISON
# ============================================================

regression_comparison = pd.DataFrame({

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
        baseline_mae,
        mae,
        tree_mae,
        rf_mae,
        gb_mae,
        svr_mae,
        best_rf_mae
    ],

    "RMSE": [
        baseline_rmse,
        rmse,
        tree_rmse,
        rf_rmse,
        gb_rmse,
        svr_rmse,
        best_rf_rmse
    ],

    "R2": [
        baseline_r2,
        r2,
        tree_r2,
        rf_r2,
        gb_r2,
        svr_r2,
        best_rf_r2
    ]
})


print("\n===== REGRESSION MODEL COMPARISON =====")

print(
    regression_comparison
    .round(3)
    .to_string(index=False)
)


# ============================================================
# 40. BEST REGRESSION MODEL
# ============================================================

best_by_mae = (
    regression_comparison
    .loc[
        regression_comparison["MAE"].idxmin(),
        "Model"
    ]
)

best_by_rmse = (
    regression_comparison
    .loc[
        regression_comparison["RMSE"].idxmin(),
        "Model"
    ]
)

best_by_r2 = (
    regression_comparison
    .loc[
        regression_comparison["R2"].idxmax(),
        "Model"
    ]
)


print("\n===== BEST REGRESSION MODEL =====")

print(
    "Best by MAE:",
    best_by_mae
)

print(
    "Best by RMSE:",
    best_by_rmse
)

print(
    "Best by R²:",
    best_by_r2
)


# ============================================================
# 41. FINAL SUMMARY
# ============================================================

print("\n")
print("================================================")
print("PROJECT TRAINING COMPLETED SUCCESSFULLY")
print("================================================")

print(
    "\nRegression model:",
    best_by_r2
)

print(
    "Regression MAE:",
    round(best_rf_mae, 4)
)

print(
    "Regression RMSE:",
    round(best_rf_rmse, 4)
)

print(
    "Regression R²:",
    round(best_rf_r2, 4)
)

print(
    "\nClassification model:",
    best_classification_model
)

print(
    "Classification Accuracy:",
    round(svm_accuracy, 4)
)

print(
    "Classification F1:",
    round(svm_f1, 4)
)

print(
    "Classification ROC-AUC:",
    round(svm_roc_auc, 4)
)

print("\n================================================")