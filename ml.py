import pickle
import numpy as np
import pandas as pd


# ==========================================
# LOAD TRAINED MODEL AND SCALER
# ==========================================

with open("models/best_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("models/scaler.pkl", "rb") as file:
    scaler = pickle.load(file)


# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_diabetes(data):

    # Convert input data into NumPy array
    input_data = pd.DataFrame([data], columns=[
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
])

    # Scale input using the same scaler used during training
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)[0]

    # Return result
    if prediction == 1:
        return "High Risk of Diabetes"
    else:
        return "Low Risk of Diabetes"
if __name__ == "__main__":

    test_data = [
        2,      # Pregnancies
        120,    # Glucose
        70,     # Blood Pressure
        25,     # Skin Thickness
        80,     # Insulin
        28.5,   # BMI
        0.35,   # Diabetes Pedigree
        35      # Age
    ]

    result = predict_diabetes(test_data)

    print("Prediction:", result)    