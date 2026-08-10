# DiaShield 🩺

DiaShield is a machine learning-based web application that predicts the risk of diabetes based on health-related information provided by the user.

## 🌐 Live Demo

https://diashield.onrender.com/

## 🚀 Features

- Diabetes risk prediction using Machine Learning
- Logistic Regression, KNN, and SVM models
- User-friendly web interface
- Responsive design for desktop and mobile
- Input guidance for health parameters
- Low Risk / High Risk prediction
- Flask-based backend
- Deployed on Render

## 🧠 Machine Learning

The project uses the following algorithms:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)

The models are trained using the **Pima Indians Diabetes Dataset**.

The best-performing model is selected for the final prediction system.

## 📋 Input Parameters

The application takes the following inputs:

1. Number of Pregnancies
2. Glucose Level
3. Blood Pressure
4. Skin Thickness
5. Insulin
6. BMI
7. Diabetes Pedigree Function
8. Age

## 🛠️ Technologies Used

- Python
- Flask
- HTML
- CSS
- NumPy
- Pandas
- Scikit-learn
- Pickle
- Git & GitHub
- Render

## 📂 Project Structure

```text
DiaShield/
│
├── app.py
├── ml.py
├── train_model.py
├── diabetes.csv
├── requirements.txt
├── LICENSE
├── README.md
│
├── models/
├── static/
└── templates/
