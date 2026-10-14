# ============================================================
# CATEGORY CONSISTENCY CHECK
# ============================================================

import pandas as pd

df = pd.read_csv(
    "data/student-mat.csv",
    sep=";"
)

print("===== CATEGORY CONSISTENCY CHECK =====")

categorical_columns = df.select_dtypes(
    include=["object"]
).columns

for column in categorical_columns:

    print("\n" + column)

    print(
        df[column]
        .value_counts()
        .to_string()
    )


print("\n===== CATEGORY CHECK COMPLETED =====")