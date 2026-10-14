import pandas as pd

# Load dataset
df = pd.read_csv("data/student-mat.csv", sep=";")

# Target variable
y = df["G3"]

# Features
X = df.drop(columns=["G3"])

print("===== FEATURE ENGINEERING =====")
print("Feature shape:", X.shape)
print("Target shape:", y.shape)

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print(y.name)
from sklearn.model_selection import train_test_split

# Split data into training and testing sets
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
# Identify numerical and categorical columns

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("\n===== FEATURE TYPES =====")

print("\nNumerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)

print("\nNumber of numerical features:", len(numerical_features))
print("Number of categorical features:", len(categorical_features))
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Create preprocessing transformer
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)

# Fit only on training data
X_train_processed = preprocessor.fit_transform(X_train)

# Transform test data
X_test_processed = preprocessor.transform(X_test)

print("\n===== PREPROCESSING =====")
print("Original training shape:", X_train.shape)
print("Processed training shape:", X_train_processed.shape)

print("Original testing shape:", X_test.shape)
print("Processed testing shape:", X_test_processed.shape)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train_processed, y_train)

# Make predictions
y_pred = model.predict(X_test_processed)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n===== LINEAR REGRESSION RESULTS =====")
print("MAE:", mae)
print("RMSE:", rmse)
print("R² Score:", r2)
from sklearn.tree import DecisionTreeRegressor

# Create Decision Tree model
tree_model = DecisionTreeRegressor(
    random_state=42,
    max_depth=5
)

# Train
tree_model.fit(X_train_processed, y_train)

# Predict
tree_pred = tree_model.predict(X_test_processed)

# Evaluate
tree_mae = mean_absolute_error(y_test, tree_pred)
tree_rmse = np.sqrt(mean_squared_error(y_test, tree_pred))
tree_r2 = r2_score(y_test, tree_pred)

print("\n===== DECISION TREE RESULTS =====")
print("MAE:", tree_mae)
print("RMSE:", tree_rmse)
print("R² Score:", tree_r2)
from sklearn.ensemble import RandomForestRegressor

# Create Random Forest model
rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

# Train
rf_model.fit(X_train_processed, y_train)

# Predict
rf_pred = rf_model.predict(X_test_processed)

# Evaluate
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print("\n===== RANDOM FOREST RESULTS =====")
print("MAE:", rf_mae)
print("RMSE:", rf_rmse)
print("R² Score:", rf_r2)
from sklearn.ensemble import GradientBoostingRegressor

# Create Gradient Boosting model
gb_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

# Train
gb_model.fit(X_train_processed, y_train)

# Predict
gb_pred = gb_model.predict(X_test_processed)

# Evaluate
gb_mae = mean_absolute_error(y_test, gb_pred)
gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
gb_r2 = r2_score(y_test, gb_pred)

print("\n===== GRADIENT BOOSTING RESULTS =====")
print("MAE:", gb_mae)
print("RMSE:", gb_rmse)
print("R² Score:", gb_r2)
from sklearn.svm import SVR

# Create SVR model
svr_model = SVR(
    kernel="rbf",
    C=10,
    epsilon=0.1
)

# Train
svr_model.fit(X_train_processed, y_train)

# Predict
svr_pred = svr_model.predict(X_test_processed)

# Evaluate
svr_mae = mean_absolute_error(y_test, svr_pred)
svr_rmse = np.sqrt(mean_squared_error(y_test, svr_pred))
svr_r2 = r2_score(y_test, svr_pred)

print("\n===== SVR RESULTS =====")
print("MAE:", svr_mae)
print("RMSE:", svr_rmse)
print("R² Score:", svr_r2)
from sklearn.model_selection import RandomizedSearchCV

# Random Forest hyperparameter tuning
rf_base = RandomForestRegressor(
    random_state=42
)

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10, 15, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": [0.5, 0.75, 1.0]
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

# Perform tuning
rf_search.fit(X_train_processed, y_train)

print("\n===== RANDOM FOREST TUNING =====")
print("Best parameters:")
print(rf_search.best_params_)

print("\nBest CV MAE:")
print(-rf_search.best_score_)
# Evaluate the best tuned Random Forest on the test set

best_rf = rf_search.best_estimator_

best_rf_pred = best_rf.predict(X_test_processed)

best_rf_mae = mean_absolute_error(y_test, best_rf_pred)
best_rf_rmse = np.sqrt(mean_squared_error(y_test, best_rf_pred))
best_rf_r2 = r2_score(y_test, best_rf_pred)

print("\n===== TUNED RANDOM FOREST TEST RESULTS =====")
print("MAE:", best_rf_mae)
print("RMSE:", best_rf_rmse)
print("R² Score:", best_rf_r2)
# Create support groups based on final grade (G3)

def support_group(score):
    if score < 10:
        return "High Support"
    elif score < 15:
        return "Medium Support"
    else:
        return "Low Support"


df["Support_Level"] = df["G3"].apply(support_group)

print("\n===== SUPPORT GROUP DISTRIBUTION =====")
print(df["Support_Level"].value_counts())
# Classification features and target

X_class = df.drop(columns=["G3", "Support_Level"])
y_class = df["Support_Level"]

print("\n===== CLASSIFICATION DATA =====")
print("Features shape:", X_class.shape)
print("Target shape:", y_class.shape)

print("\nTarget classes:")
print(y_class.value_counts())
from sklearn.model_selection import train_test_split

X_class_train, X_class_test, y_class_train, y_class_test = train_test_split(
    X_class,
    y_class,
    test_size=0.20,
    random_state=42,
    stratify=y_class
)

print("\n===== CLASSIFICATION TRAIN TEST SPLIT =====")
print("Training features:", X_class_train.shape)
print("Testing features:", X_class_test.shape)
print("Training target:", y_class_train.shape)
print("Testing target:", y_class_test.shape)

print("\nTraining class distribution:")
print(y_class_train.value_counts())

print("\nTesting class distribution:")
print(y_class_test.value_counts())
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Identify feature types
class_numerical_features = X_class_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

class_categorical_features = X_class_train.select_dtypes(
    include=["object"]
).columns.tolist()

# Preprocessing
class_preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
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
print("Original training shape:", X_class_train.shape)
print("Processed training shape:", X_class_train_processed.shape)
print("Original testing shape:", X_class_test.shape)
print("Processed testing shape:", X_class_test_processed.shape)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

accuracy = accuracy_score(y_class_test, logistic_pred)

precision = precision_score(
    y_class_test,
    logistic_pred,
    average="weighted"
)

recall = recall_score(
    y_class_test,
    logistic_pred,
    average="weighted"
)

f1 = f1_score(
    y_class_test,
    logistic_pred,
    average="weighted"
)

print("\n===== LOGISTIC REGRESSION RESULTS =====")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(
    y_class_test,
    logistic_pred,
    labels=["High Support", "Medium Support", "Low Support"]
)

print("\n===== CONFUSION MATRIX =====")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["High Support", "Medium Support", "Low Support"]
)

disp.plot()
plt.title("Logistic Regression - Confusion Matrix")
plt.show()
from sklearn.tree import DecisionTreeClassifier

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

dt_accuracy = accuracy_score(y_class_test, decision_tree_pred)

dt_precision = precision_score(
    y_class_test,
    decision_tree_pred,
    average="weighted"
)

dt_recall = recall_score(
    y_class_test,
    decision_tree_pred,
    average="weighted"
)

dt_f1 = f1_score(
    y_class_test,
    decision_tree_pred,
    average="weighted"
)

print("\n===== DECISION TREE RESULTS =====")
print("Accuracy:", dt_accuracy)
print("Precision:", dt_precision)
print("Recall:", dt_recall)
print("F1 Score:", dt_f1)
from sklearn.ensemble import RandomForestClassifier

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

rf_accuracy = accuracy_score(y_class_test, random_forest_pred)

rf_precision = precision_score(
    y_class_test,
    random_forest_pred,
    average="weighted"
)

rf_recall = recall_score(
    y_class_test,
    random_forest_pred,
    average="weighted"
)

rf_f1 = f1_score(
    y_class_test,
    random_forest_pred,
    average="weighted"
)

print("\n===== RANDOM FOREST CLASSIFIER RESULTS =====")
print("Accuracy:", rf_accuracy)
print("Precision:", rf_precision)
print("Recall:", rf_recall)
print("F1 Score:", rf_f1)
from sklearn.ensemble import GradientBoostingClassifier

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

gb_accuracy = accuracy_score(y_class_test, gradient_pred)

gb_precision = precision_score(
    y_class_test,
    gradient_pred,
    average="weighted"
)

gb_recall = recall_score(
    y_class_test,
    gradient_pred,
    average="weighted"
)

gb_f1 = f1_score(
    y_class_test,
    gradient_pred,
    average="weighted"
)

print("\n===== GRADIENT BOOSTING CLASSIFIER RESULTS =====")
print("Accuracy:", gb_accuracy)
print("Precision:", gb_precision)
print("Recall:", gb_recall)
print("F1 Score:", gb_f1)
from sklearn.svm import SVC

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

svm_accuracy = accuracy_score(y_class_test, svm_pred)

svm_precision = precision_score(
    y_class_test,
    svm_pred,
    average="weighted"
)

svm_recall = recall_score(
    y_class_test,
    svm_pred,
    average="weighted"
)

svm_f1 = f1_score(
    y_class_test,
    svm_pred,
    average="weighted"
)

print("\n===== SVM CLASSIFIER RESULTS =====")
print("Accuracy:", svm_accuracy)
print("Precision:", svm_precision)
print("Recall:", svm_recall)
print("F1 Score:", svm_f1)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

svm_cm = confusion_matrix(
    y_class_test,
    svm_pred,
    labels=["High Support", "Medium Support", "Low Support"]
)

print("\n===== SVM CONFUSION MATRIX =====")
print(svm_cm)

svm_disp = ConfusionMatrixDisplay(
    confusion_matrix=svm_cm,
    display_labels=["High Support", "Medium Support", "Low Support"]
)

svm_disp.plot()
plt.title("SVM - Support Level Confusion Matrix")
plt.show()
# ===== CLUSTERING DATA =====

X_cluster = df.drop(columns=["G3", "Support_Level"])

print("\n===== CLUSTERING DATA =====")
print("Clustering features shape:", X_cluster.shape)

print("\nFeatures used for clustering:")
print(X_cluster.columns.tolist())
from sklearn.preprocessing import StandardScaler

# Identify numerical and categorical features
cluster_numerical_features = X_cluster.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

cluster_categorical_features = X_cluster.select_dtypes(
    include=["object"]
).columns.tolist()

# One-hot encode categorical features
cluster_preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            cluster_categorical_features
        )
    ],
    remainder="passthrough"
)

X_cluster_encoded = cluster_preprocessor.fit_transform(X_cluster)

# Scale all features
scaler = StandardScaler(with_mean=False)

X_cluster_scaled = scaler.fit_transform(X_cluster_encoded)

print("\n===== CLUSTERING PREPROCESSING =====")
print("Original shape:", X_cluster.shape)
print("Encoded shape:", X_cluster_encoded.shape)
print("Scaled shape:", X_cluster_scaled.shape)
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inertia_values = []

for k in range(2, 9):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_cluster_scaled)

    inertia_values.append(kmeans.inertia_)

print("\n===== ELBOW METHOD =====")

for k, inertia in zip(range(2, 9), inertia_values):
    print("K =", k, "Inertia =", inertia)

plt.figure(figsize=(8, 5))
plt.plot(range(2, 9), inertia_values, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for K-Means Clustering")
plt.grid(True)
plt.show()
# ===== K-MEANS CLUSTERING =====

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

cluster_labels = kmeans.fit_predict(X_cluster_scaled)

print("\n===== K-MEANS RESULTS =====")
print("Number of clusters:", 4)

print("\nStudents in each cluster:")
print(pd.Series(cluster_labels).value_counts().sort_index())
# ===== K-MEANS CLUSTERING =====

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

cluster_labels = kmeans.fit_predict(X_cluster_scaled)

print("\n===== K-MEANS RESULTS =====")
print("Number of clusters:", 4)

print("\nStudents in each cluster:")
print(pd.Series(cluster_labels).value_counts().sort_index())
# ===== CLUSTER PROFILING =====

cluster_profile = X_cluster.copy()
cluster_profile["Cluster"] = cluster_labels

profile = cluster_profile.groupby("Cluster")[
    ["age", "studytime", "failures", "absences"]
].mean()

print("\n===== CLUSTER PROFILE =====")
print(profile.round(2).to_string())

print("\n===== CLUSTER SIZES =====")
print(
    cluster_profile["Cluster"]
    .value_counts()
    .sort_index()
)
# ===== CLUSTER VISUALIZATION =====

plt.figure(figsize=(8, 5))

plt.scatter(
    cluster_profile["studytime"],
    cluster_profile["failures"],
    c=cluster_profile["Cluster"],
    cmap="viridis"
)

plt.xlabel("Study Time")
plt.ylabel("Previous Failures")
plt.title("Student Clusters: Study Time vs Previous Failures")
plt.colorbar(label="Cluster")
plt.show()
# ===== SAVE TRAINED MODEL AND PREPROCESSOR =====

import joblib
import os

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save tuned Random Forest model
joblib.dump(
    best_rf,
    "models/tuned_random_forest_regressor.pkl"
)

# Save preprocessing object
joblib.dump(
    preprocessor,
    "models/regression_preprocessor.pkl"
)

print("\n===== MODEL SAVING =====")
print("Tuned Random Forest saved successfully.")
print("Preprocessor saved successfully.")
# ===== REGRESSION RESIDUAL ANALYSIS =====

residuals = y_test - best_rf_pred

print("\n===== RESIDUAL ANALYSIS =====")
print("Mean residual:", residuals.mean())
print("Minimum residual:", residuals.min())
print("Maximum residual:", residuals.max())

plt.figure(figsize=(8, 5))

plt.scatter(
    best_rf_pred,
    residuals
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Final Score")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Random Forest Residual Analysis")

plt.show()
# ===== REGRESSION MODEL COMPARISON =====

comparison = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
        "SVR",
        "Tuned Random Forest"
    ],
    "MAE": [
        mae,
        tree_mae,
        rf_mae,
        gb_mae,
        svr_mae,
        best_rf_mae
    ],
    "RMSE": [
        rmse,
        tree_rmse,
        rf_rmse,
        gb_rmse,
        svr_rmse,
        best_rf_rmse
    ],
    "R2": [
        r2,
        tree_r2,
        rf_r2,
        gb_r2,
        svr_r2,
        best_rf_r2
    ]
})

print("\n===== REGRESSION MODEL COMPARISON =====")
print(comparison.round(3).to_string(index=False))

print("\n===== BEST REGRESSION MODEL =====")
print("Best by MAE:", comparison.loc[comparison["MAE"].idxmin(), "Model"])
print("Best by RMSE:", comparison.loc[comparison["RMSE"].idxmin(), "Model"])
print("Best by R²:", comparison.loc[comparison["R2"].idxmax(), "Model"])
# ===== SVM ROC-AUC =====

from sklearn.metrics import roc_auc_score

# Get predicted probabilities
svm_probabilities = svm_model.predict_proba(
    X_class_test_processed
)

# Multiclass ROC-AUC
svm_roc_auc = roc_auc_score(
    y_class_test,
    svm_probabilities,
    multi_class="ovr",
    average="weighted"
)

print("\n===== SVM ROC-AUC =====")
print("Weighted ROC-AUC:", svm_roc_auc)
# ===== CLASSIFICATION MODEL COMPARISON =====

classification_comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
        "SVM"
    ],
    "Accuracy": [
        accuracy,
        dt_accuracy,
        rf_accuracy,
        gb_accuracy,
        svm_accuracy
    ],
    "Precision": [
        precision,
        dt_precision,
        rf_precision,
        gb_precision,
        svm_precision
    ],
    "Recall": [
        recall,
        dt_recall,
        rf_recall,
        gb_recall,
        svm_recall
    ],
    "F1 Score": [
        f1,
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

print("\n===== BEST CLASSIFICATION MODEL =====")

print(
    "Best by Accuracy:",
    classification_comparison.loc[
        classification_comparison["Accuracy"].idxmax(),
        "Model"
    ]
)

print(
    "Best by F1 Score:",
    classification_comparison.loc[
        classification_comparison["F1 Score"].idxmax(),
        "Model"
    ]
)
print("\n===== CLASSIFICATION RESULTS =====")

print("Logistic Regression Accuracy:", accuracy)
print("Decision Tree Accuracy:", dt_accuracy)
print("Random Forest Accuracy:", rf_accuracy)
print("Gradient Boosting Accuracy:", gb_accuracy)
print("SVM Accuracy:", svm_accuracy)

print("\n===== BEST CLASSIFICATION MODEL =====")
print("SVM")
# ============================================================
# SAVE CLASSIFICATION MODEL AND PREPROCESSOR
# ============================================================

import joblib
import os

os.makedirs("models", exist_ok=True)

joblib.dump(
    svm_model,
    "models/svm_classifier.pkl"
)

joblib.dump(
    class_preprocessor,
    "models/classification_preprocessor.pkl"
)

print("\n===== CLASSIFICATION MODEL SAVING =====")
print("SVM classifier saved successfully.")
print("Classification preprocessor saved successfully.")