import pandas as pd

# Load dataset
file_path = "data/student-mat.csv"
df = pd.read_csv(file_path, sep=";")

print("===== DATASET SHAPE =====")
print(df.shape)

print("\n===== MISSING VALUES =====")
print(df.isnull().sum().sum())

print("\n===== DUPLICATE ROWS =====")
print(df.duplicated().sum())

print("\n===== UNIQUE VALUES IN CATEGORICAL COLUMNS =====")

categorical_columns = df.select_dtypes(include="object").columns

for column in categorical_columns:
    print(f"\n{column}:")
    print(df[column].unique())

print("\n===== NUMERICAL RANGE CHECK =====")

print("Age:", df["age"].min(), "to", df["age"].max())
print("Study time:", df["studytime"].min(), "to", df["studytime"].max())
print("Failures:", df["failures"].min(), "to", df["failures"].max())
print("Absences:", df["absences"].min(), "to", df["absences"].max())
print("G1:", df["G1"].min(), "to", df["G1"].max())
print("G2:", df["G2"].min(), "to", df["G2"].max())
print("G3:", df["G3"].min(), "to", df["G3"].max())
