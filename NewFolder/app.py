import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Load trained models
reg_model = pickle.load(open("reg_model.pkl", "rb"))
log_model = pickle.load(open("log_model.pkl", "rb"))

st.title("Student Academic Performance Predictor")
st.write("""
Enter the student details below to predict:
1. Expected Study Hours (Regression)
2. Pass/Fail Status (Classification)
""")

# User Inputs
study_hours_input = st.number_input("Study Hours", min_value=0, max_value=24, value=2)
attendance_input = st.number_input("Attendance Percentage", min_value=0, max_value=100, value=90)
parent_edu_input = st.selectbox("Parent Education", ["high school", "bachelor", "master", "post graduate"])
gender_input = st.selectbox("Gender", ["Male", "Female", "Other"])
school_input = st.selectbox("School Type", ["Public", "Private"])
internet_input = st.selectbox("Internet Access", ["Yes", "No"])
travel_time_input = st.number_input("Travel Time (minutes)", min_value=0, max_value=120, value=15)
extra_input = st.selectbox("Participation in Extra Activities", ["Yes", "No"])

# Encode categorical inputs
le_gender = LabelEncoder()
le_gender.classes_ = np.array(["Female","Male","Other"])
gender_encoded = le_gender.transform([gender_input])[0]

le_school = LabelEncoder()
le_school.classes_ = np.array(["Private","Public"])
school_encoded = le_school.transform([school_input])[0]

le_parent = LabelEncoder()
le_parent.classes_ = np.array(["bachelor","high school","master","post graduate"])
parent_encoded = le_parent.transform([parent_edu_input])[0]

le_internet = LabelEncoder()
le_internet.classes_ = np.array(["No", "Yes"])
internet_encoded = le_internet.transform([internet_input])[0]

le_extra = LabelEncoder()
le_extra.classes_ = np.array(["No", "Yes"])
extra_encoded = le_extra.transform([extra_input])[0]

# Prepare features for prediction
X_new = np.array([[study_hours_input, attendance_input, parent_encoded, gender_encoded,
                   school_encoded, internet_encoded, travel_time_input, extra_encoded]])

# Predictions
pred_score = reg_model.predict(X_new)[0]
pred_class = log_model.predict(X_new)[0]

st.write(f"**Predicted Study Hours (Regression):** {pred_score:.2f}")
st.write(f"**Predicted Pass/Fail (Classification):** {'Pass' if pred_class==1 else 'Fail'}")