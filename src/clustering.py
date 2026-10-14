import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load dataset
df = pd.read_csv("data/student-mat.csv", sep=";")

# Remove target column
# G3 must NOT be used for clustering
X = df.drop(columns=["G3"])

# Select numerical features
X_numeric = X.select_dtypes(include=["int64", "float64"])

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

# Create final K-Means model
kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

# Assign clusters
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Calculate silhouette score
score = silhouette_score(X_scaled, df["Cluster"])

print("===== STUDENT CLUSTERING =====")
print(f"Number of clusters: 2")
print(f"Silhouette Score: {score:.4f}")

# Cluster sizes
print("\nCluster Sizes:")
print(df["Cluster"].value_counts().sort_index())

# Average numerical characteristics
print("\nCluster Characteristics:")
cluster_summary = df.groupby("Cluster")[X_numeric.columns].mean().round(2)
print(cluster_summary)

# Visualization
plt.figure(figsize=(8, 5))

plt.scatter(
    df["studytime"],
    df["absences"],
    c=df["Cluster"]
)

plt.xlabel("Study Time")
plt.ylabel("Absences")
plt.title("Student Clusters: Study Time vs Absences")

plt.show()