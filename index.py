import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score
import pickle

# Load dataset
df = pd.read_csv("student_performance.csv")

# Encode categorical variables
le_gender = LabelEncoder()
df['gender'] = le_gender.fit_transform(df['gender'])

le_school = LabelEncoder()
df['school_type'] = le_school.fit_transform(df['school_type'])

le_parent = LabelEncoder()
df['parent_education'] = le_parent.fit_transform(df['parent_education'])

le_internet = LabelEncoder()
df['internet_access'] = le_internet.fit_transform(df['internet_access'])

le_extra = LabelEncoder()
df['extra_activities'] = le_extra.fit_transform(df['extra_activities'])

# Map travel_time to numeric minutes
travel_map = {
    "<15 min": 10,
    "15-30 min": 22,
    "30-60 min": 45,
    ">60 min": 75
}
df['travel_time'] = df['travel_time'].map(travel_map)

# Features
X = df[['study_hours', 'attendance_percentage', 'parent_education',
        'gender', 'school_type', 'internet_access', 'travel_time', 'extra_activities']]

# Regression target
y_score = df['study_hours']  # or overall_score if available

# Classification target
df['pass_fail'] = np.where(df['study_hours'] >= 2, 1, 0)
y_grade = df['pass_fail']

# Split data
X_train, X_test, y_train_score, y_test_score = train_test_split(X, y_score, test_size=0.2, random_state=42)
_, _, y_train_grade, y_test_grade = train_test_split(X, y_grade, test_size=0.2, random_state=42)

# Train models
reg_model = LinearRegression()
reg_model.fit(X_train, y_train_score)

log_model = LogisticRegression(max_iter=300)
log_model.fit(X_train, y_train_grade)

# Test models
y_pred_score = reg_model.predict(X_test)
y_pred_grade = log_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test_score, y_pred_score))
accuracy = accuracy_score(y_test_grade, y_pred_grade)

print("Regression RMSE:", rmse)
print("Classification Accuracy:", accuracy)

# Save models
pickle.dump(reg_model, open("reg_model.pkl", "wb"))
pickle.dump(log_model, open("log_model.pkl", "wb"))

print("Models saved successfully.")