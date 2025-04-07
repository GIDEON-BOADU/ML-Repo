# src/train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Load dataset
df = pd.read_csv("D:/AI-project/heart_disease_predictor/data/heart.csv")

# Split features and target
X = df.drop("target", axis=1)
y = df["target"]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model and scaler
os.makedirs("../models", exist_ok=True)
joblib.dump(model, "../models/heart_disease_model.pkl")
joblib.dump(scaler, "../models/scaler.pkl")
print("Model and scaler saved successfully!")
