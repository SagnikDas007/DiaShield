
# DiaShield 🩺

**DiaShield** is a robust machine learning-powered web application designed to predict the risk of diabetes based on specific health parameters. By leveraging advanced data science algorithms, it provides users with an instant assessment of their health risk levels (High Risk vs. Low Risk).

🌐 **[Live Demo](https://diashield.onrender.com/)**

---

## 🚀 Features

- **Accurate Predictions:** Uses trained Machine Learning models to analyze health data.
- **Multiple Algorithm Support:** Implements Logistic Regression, K-Nearest Neighbors (KNN), and Support Vector Machine (SVM) to find the most accurate result.
- **User-Friendly Interface:** Clean, intuitive, and responsive UI built for both desktop and mobile devices.
- **Instant Results:** Real-time processing of inputs with immediate risk classification.
- **Input Guidance:** Clear labels and descriptions for required health parameters.

---

## 🧠 Machine Learning Overview

DiaShield uses the **Pima Indians Diabetes Dataset** to train its models. The system evaluates multiple classification algorithms to ensure reliable output:

- **Logistic Regression:** For baseline statistical probability.
- **K-Nearest Neighbors (KNN):** To identify patterns based on data similarity.
- **Support Vector Machine (SVM):** For high-dimensional classification accuracy.

The models are serialized using `Pickle` and served through a Flask backend.

---

## 📋 Input Parameters

To get a prediction, users provide the following 8 parameters:
1. **Pregnancies:** Number of times pregnant.
2. **Glucose:** Plasma glucose concentration.
3. **Blood Pressure:** Diastolic blood pressure (mm Hg).
4. **Skin Thickness:** Triceps skin fold thickness (mm).
5. **Insulin:** 2-Hour serum insulin (mu U/ml).
6. **BMI:** Body Mass Index (weight in kg/(height in m)²).
7. **Pedigree Function:** Diabetes pedigree function (genetic history).
8. **Age:** Age in years.

---

## 🛠️ Technologies Used

- **Backend:** [Flask](https://flask.palletsprojects.com/) (Python)
- **Machine Learning:** [Scikit-learn](https://scikit-learn.org/), [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Frontend:** HTML5, CSS3 (Modern UI)
- **Deployment:** [Render](https://render.com/)
- **Version Control:** Git & GitHub

---

## 📂 Project Structure

```text
DiaShield/
│
├── app.py              # Main Flask application
├── ml.py               # Machine Learning logic and helper functions
├── train_model.py      # Script to train and save the model
├── diabetes.csv        # Dataset used for training
├── requirements.txt    # List of Python dependencies
├── LICENSE             # MIT License
├── README.md           # Documentation
│
├── models/             # Saved .pkl model files
├── static/             # CSS, Images, and JavaScript files
└── templates/          # HTML templates (index.html, result.html)
```

---

## ⚙️ Installation & Setup

If you want to run this project locally, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/SagnikDas007/DiaShield.git
   cd DiaShield
   ```

2. **Create a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the Model (Optional):**
   If the `models/` folder is empty, generate the model files by running:
   ```bash
   python train_model.py
   ```

5. **Run the Application:**
   ```bash
   python app.py
   ```
   Open `http://127.0.0.1:5000/` in your browser.

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! If you have ideas for improvement or find any bugs, feel free to open an issue or submit a pull request.

1. Fork the project.
2. Create your feature branch (`git checkout -b feature/NewFeature`).
3. Commit your changes (`git commit -m 'Add some NewFeature'`).
4. Push to the branch (`git push origin feature/NewFeature`).
5. Open a Pull Request.

---

## ✉️ Contact

**Sagnik Das**  
GitHub: [@SagnikDas007](https://github.com/SagnikDas007)  
Project Link: [https://github.com/SagnikDas007/DiaShield](https://github.com/SagnikDas007/DiaShield)

---
*Disclaimer: This tool is for educational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment.*
