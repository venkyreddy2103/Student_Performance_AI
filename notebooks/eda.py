import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/student-mat.csv", sep=";")

# Plot G3 distribution
plt.figure(figsize=(8, 5))

plt.hist(df["G3"], bins=21)

plt.title("Distribution of Final Grade (G3)")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Number of Students")

plt.show()
# Visualization 2: Study Time vs Final Grade

plt.figure(figsize=(8, 5))

plt.scatter(df["studytime"], df["G3"])

plt.title("Study Time vs Final Grade")
plt.xlabel("Study Time")
plt.ylabel("Final Grade (G3)")

plt.show()
# Visualization 3: Failures vs Final Grade

plt.figure(figsize=(8, 5))

plt.scatter(df["failures"], df["G3"])

plt.title("Previous Failures vs Final Grade")
plt.xlabel("Number of Previous Failures")
plt.ylabel("Final Grade (G3)")

plt.show()
# Visualization 4: Gender vs Average Final Grade

gender_grade = df.groupby("sex")["G3"].mean()

plt.figure(figsize=(8, 5))

plt.bar(gender_grade.index, gender_grade.values)

plt.title("Average Final Grade by Gender")
plt.xlabel("Gender")
plt.ylabel("Average Final Grade (G3)")
# Visualization 5: Absences vs Final Grade

plt.figure(figsize=(8, 5))

plt.scatter(df["absences"], df["G3"])

plt.title("Absences vs Final Grade")
plt.xlabel("Number of Absences")
plt.ylabel("Final Grade (G3)")
# Visualization 6: School vs Average Final Grade

school_grade = df.groupby("school")["G3"].mean()

plt.figure(figsize=(8, 5))

plt.bar(school_grade.index, school_grade.values)

plt.title("Average Final Grade by School")
plt.xlabel("School")
plt.ylabel("Average Final Grade (G3)")
plt.show()
# Visualization 7: Study Time Distribution

study_counts = df["studytime"].value_counts().sort_index()

plt.figure(figsize=(8, 5))

plt.bar(study_counts.index, study_counts.values)

plt.title("Distribution of Study Time")
plt.xlabel("Study Time")
plt.ylabel("Number of Students")
plt.show()
# Visualization 8: Internet Access vs Final Grade

internet_grade = df.groupby("internet")["G3"].mean()

plt.figure(figsize=(8, 5))

plt.bar(internet_grade.index, internet_grade.values)

plt.title("Average Final Grade by Internet Access")
plt.xlabel("Internet Access")
plt.ylabel("Average Final Grade (G3)")

