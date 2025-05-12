import pandas as pd
import joblib

# Load model, encoders, and feature list
model = joblib.load("attrition_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

with open("model_features.pkl", "rb") as f:
    feature_names = joblib.load(f)

# Example new input (replace these with actual values or use a form)
new_data = {
    "Age": 34,
    "BusinessTravel": "Travel_Rarely",
    "DailyRate": 800,
    "Department": "Research & Development",
    "DistanceFromHome": 10,
    "Education": 3,
    "EducationField": "Life Sciences",
    "EmployeeCount": 1,
    "EmployeeNumber": 9999,
    "EnvironmentSatisfaction": 3,
    "Gender": "Male",
    "HourlyRate": 60,
    "JobInvolvement": 3,
    "JobLevel": 2,
    "JobRole": "Research Scientist",
    "JobSatisfaction": 4,
    "MaritalStatus": "Single",
    "MonthlyIncome": 5000,
    "MonthlyRate": 12000,
    "NumCompaniesWorked": 1,
    "Over18": "Y",
    "OverTime": "No",
    "PercentSalaryHike": 15,
    "PerformanceRating": 3,
    "RelationshipSatisfaction": 2,
    "StandardHours": 80,
    "StockOptionLevel": 1,
    "TotalWorkingYears": 10,
    "TrainingTimesLastYear": 3,
    "WorkLifeBalance": 3,
    "YearsAtCompany": 5,
    "YearsInCurrentRole": 3,
    "YearsSinceLastPromotion": 1,
    "YearsWithCurrManager": 2
}

# Convert to DataFrame
new_df = pd.DataFrame([new_data])

# Apply label encoding
for col, le in label_encoders.items():
    if col in new_df.columns:
        new_df[col] = le.transform(new_df[col])

# Reorder columns
new_df = new_df[feature_names]

# Make prediction
prediction = model.predict(new_df)[0]
prediction_label = "Yes" if prediction == 1 else "No"

print(f"Predicted Attrition: {prediction_label}")

