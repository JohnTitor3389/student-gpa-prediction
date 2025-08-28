import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ================================
# Student GPA Prediction Project
# ================================

# 1. Load Dataset
df = pd.read_csv("synthetic_student_performance.csv")
print("Dataset Overview:\n", df.head(), "\n")

# 2. Features & Target
X = df[[
    "Age", "Gender", "Ethnicity", "ParentalEducation",
    "StudyTimeWeekly", "Absences", "Tutoring", "ParentalSupport",
    "Extracurricular", "Sports", "Music", "Volunteering", "GradeClass"
]]
y = df["GPA"]

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=73
)

# 4. Build Model
model = RandomForestRegressor(
    n_estimators=500,
    max_depth=10,
    random_state=73
)

# 5. Train Model
model.fit(X_train, y_train)

# 6. Predict
y_pred = model.predict(X_test)
print("Test Data Predictions:\n", y_pred, "\n")

# 7. Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"MSE: {mse:.5f}")
print(f"RMSE: {rmse:.5f}\n")

# 8. Feature Importance
importance = model.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
plt.barh(features, importance, color="skyblue")
plt.xlabel("Feature Importance")
plt.title("Random Forest - Feature Importance")
plt.gca().invert_yaxis()
plt.show()


# ================================
# Notes
# ================================
# 🤝 Collaboration Note
# This project is the result of a collaboration between human creativity and AI assistance.
# Human (Jonathan Mike Frenky) designed the project idea, selected the dataset, and built the core logic (data selection, model choice, evaluation metric).
# AI Assistant (ChatGPT) helped refine and clean the code structure, improved readability and added interpretability (feature importance visualization).
# By combining human intuition with AI support, the project demonstrates a future-ready workflow for data science where humans focus on ideas and problem-solving, while AI supports efficiency and presentation.


# 📂 Column Description:

# 🎓 Student Information
# StudentID → Unique identifier for each student (1001 to 6000).

# 👤 Demographic Details
# Age → Student’s age (15–18).
# Gender → 0 = Male, 1 = Female
# Ethnicity → 0 = Caucasian, 1 = African American, 2 = Asian, 3 = Other
# ParentalEducation → 0 = None, 1 = High School, 2 = Some College, 3 = Bachelor’s, 4 = Higher

# 📚 Study Habits
# StudyTimeWeekly → Weekly study time (hours, 0–20).
# Absences → Number of school absences (0–30).
# Tutoring → 0 = No, 1 = Yes

# 👨‍👩‍👧 Parental Involvement
# ParentalSupport → 0 = None, 1 = Low, 2 = Moderate, 3 = High, 4 = Very High

# 🎭 Extracurricular Activities
# Extracurricular → 0 = No, 1 = Yes
# Sports → 0 = No, 1 = Yes
# Music → 0 = No, 1 = Yes
# Volunteering → 0 = No, 1 = Yes

# 🏆 Academic Performance
# GPA → Grade Point Average (scale 2.0–4.0).
# GradeClass → 0 = A, 1 = B, 2 = C, 3 = D, 4 = F

# 📂 Dataset Source
# Kaggle: Synthetic Student Performance Dataset
# https://www.kaggle.com/datasets/miadul/student-performance-dataset