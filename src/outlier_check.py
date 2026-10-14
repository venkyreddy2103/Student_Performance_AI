# ============================================================
# OUTLIER DETECTION USING IQR
# ============================================================

import pandas as pd

# Load dataset
df = pd.read_csv(
    "data/student-mat.csv",
    sep=";"
)

print("===== OUTLIER DETECTION =====")
print("Dataset shape:", df.shape)


# Select numerical columns
numerical_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns


print("\nNumerical columns:")
print(list(numerical_columns))


# ============================================================
# IQR OUTLIER CHECK
# ============================================================

outlier_results = []

for column in numerical_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ]

    outlier_count = len(outliers)

    outlier_results.append({
        "Feature": column,
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "Lower Bound": lower_bound,
        "Upper Bound": upper_bound,
        "Outlier Count": outlier_count
    })


# Create results table
results = pd.DataFrame(outlier_results)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n===== OUTLIER RESULTS =====")

print(
    results[
        [
            "Feature",
            "Outlier Count"
        ]
    ].to_string(index=False)
)


# ============================================================
# FEATURES WITH OUTLIERS
# ============================================================

print("\n===== FEATURES WITH OUTLIERS =====")

features_with_outliers = results[
    results["Outlier Count"] > 0
]

if len(features_with_outliers) == 0:

    print("No outliers detected.")

else:

    print(
        features_with_outliers[
            [
                "Feature",
                "Outlier Count"
            ]
        ].to_string(index=False)
    )


# ============================================================
# TOTAL OUTLIERS
# ============================================================

total_outliers = results["Outlier Count"].sum()

print(
    "\nTotal outlier observations:",
    total_outliers
)

print("\n===== OUTLIER CHECK COMPLETED =====")