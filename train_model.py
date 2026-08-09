import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("diabetes.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ==========================================
# 2. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop("Outcome", axis=1)
y = df["Outcome"]


# ==========================================
# 3. HANDLE INVALID ZERO VALUES
# ==========================================

columns_with_zero = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

X[columns_with_zero] = X[columns_with_zero].replace(0, np.nan)

# Replace missing values with median
X[columns_with_zero] = X[columns_with_zero].fillna(
    X[columns_with_zero].median()
)


# ==========================================
# 4. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 5. FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==========================================
# 6. CREATE MODELS
# ==========================================

lr_model = LogisticRegression(max_iter=1000)

knn_model = KNeighborsClassifier(n_neighbors=5)

svm_model = SVC(kernel="rbf", probability=True)


# ==========================================
# 7. TRAIN MODELS
# ==========================================

print("\nTraining models...")

lr_model.fit(X_train_scaled, y_train)
knn_model.fit(X_train_scaled, y_train)
svm_model.fit(X_train_scaled, y_train)

print("All models trained successfully!")


# ==========================================
# 8. MAKE PREDICTIONS
# ==========================================

lr_prediction = lr_model.predict(X_test_scaled)
knn_prediction = knn_model.predict(X_test_scaled)
svm_prediction = svm_model.predict(X_test_scaled)


# ==========================================
# 9. CALCULATE ACCURACY
# ==========================================

lr_accuracy = accuracy_score(y_test, lr_prediction)
knn_accuracy = accuracy_score(y_test, knn_prediction)
svm_accuracy = accuracy_score(y_test, svm_prediction)


print("\n================================")
print("MODEL PERFORMANCE")
print("================================")

print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")
print(f"KNN Accuracy:                 {knn_accuracy:.4f}")
print(f"SVM Accuracy:                 {svm_accuracy:.4f}")


# ==========================================
# 10. FIND BEST MODEL
# ==========================================

models = {
    "Logistic Regression": (lr_model, lr_accuracy),
    "KNN": (knn_model, knn_accuracy),
    "SVM": (svm_model, svm_accuracy)
}

best_model_name = max(
    models,
    key=lambda model: models[model][1]
)

best_model = models[best_model_name][0]
best_accuracy = models[best_model_name][1]


print("\n================================")
print("BEST MODEL")
print("================================")

print("Best Model:", best_model_name)
print(f"Best Accuracy: {best_accuracy:.4f}")


# ==========================================
# 11. CREATE MODELS DIRECTORY
# ==========================================

os.makedirs("models", exist_ok=True)


# ==========================================
# 12. SAVE MODELS
# ==========================================

with open("models/lr_model.pkl", "wb") as file:
    pickle.dump(lr_model, file)

with open("models/knn_model.pkl", "wb") as file:
    pickle.dump(knn_model, file)

with open("models/svm_model.pkl", "wb") as file:
    pickle.dump(svm_model, file)

with open("models/scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

with open("models/best_model.pkl", "wb") as file:
    pickle.dump(best_model, file)


# ==========================================
# 13. SAVE BEST MODEL NAME
# ==========================================

with open("models/best_model_name.txt", "w") as file:
    file.write(best_model_name)


print("\n================================")
print("FILES SAVED SUCCESSFULLY")
print("================================")

print("models/lr_model.pkl")
print("models/knn_model.pkl")
print("models/svm_model.pkl")
print("models/scaler.pkl")
print("models/best_model.pkl")
print("models/best_model_name.txt")