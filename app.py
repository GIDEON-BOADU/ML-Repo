from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and scaler
model = joblib.load("heart_disease_predictor/models/models/heart_disease_model.pkl")
scaler = joblib.load("heart_disease_predictor/models/models/scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get form data
    age = int(request.form["age"])
    sex = int(request.form["sex"])
    chest_pain_type = int(request.form["chest_pain_type"])
    resting_bp = int(request.form["resting_bp"])
    chol = int(request.form["chol"])
    fasting_blood_sugar = int(request.form["fasting_blood_sugar"])
    restecg = int(request.form["restecg"])
    max_hr = int(request.form["max_hr"])
    exang = int(request.form["exang"])
    oldpeak = float(request.form["oldpeak"])
    slope = int(request.form["slope"])
    num_major_vessels = int(request.form["num_major_vessels"])
    thal = int(request.form["thal"])

    # Combine all inputs into a single array
    input_features = np.array([[age, sex, chest_pain_type, resting_bp, chol, 
                                fasting_blood_sugar, restecg, max_hr, exang, 
                                oldpeak, slope, num_major_vessels, thal]])

    # Scale the features
    scaled_features = scaler.transform(input_features)

    # Make prediction
    prediction = model.predict(scaled_features)

    # Display prediction result
    prediction_text = "Positive for heart disease" if prediction[0] == 1 else "No heart disease"
    
    return render_template("index.html", prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
