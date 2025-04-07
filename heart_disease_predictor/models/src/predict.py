# src/predict.py

import joblib
import numpy as np

# Load model and scaler
model = joblib.load("../models/heart_disease_model.pkl")
scaler = joblib.load("../models/scaler.pkl")

def predict_heart_disease(input_data):
    """
    input_data: list or array of 13 features in correct order
    returns: prediction (0 = no disease, 1 = disease)
    """
    input_scaled = scaler.transform([input_data])
    prediction = model.predict(input_scaled)
    return prediction[0]

# Example: Predict for a new patient
# Format: [age, sex, chest_pain_type, resting_bp, cholesterol,
#          fasting_blood_sugar, restecg, max_hr, exang,
#          oldpeak, slope, num_major_vessels, thal]

example = [62, 1, 0, 140, 268, 0, 1, 160, 0, 3.6, 2, 2, 2]
result = predict_heart_disease(example)
print("Prediction:", "Heart Disease" if result == 1 else "No Heart Disease")
