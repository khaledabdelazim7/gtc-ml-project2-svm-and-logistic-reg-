Diabetes Classification with SVM & Logistic Regression

This project demonstrates how to classify diabetes cases using Support Vector Machine (SVM) and Logistic Regression with Python’s scikit-learn.
It includes data loading, preprocessing, model training, and evaluation steps.

📂 Project Structure

Diabetes_khaled_abdelazim.ipynb → Jupyter notebook containing the implementation.

README.md → Documentation for setup and usage.

🚀 Getting Started
1️⃣ Install Dependencies

Make sure you have Python 3.8+ and install the required libraries:

pip install numpy pandas scikit-learn matplotlib seaborn

2️⃣ Run the Notebook

Start Jupyter Notebook and open the file:

jupyter notebook Diabetes_khaled_abdelazim.ipynb

🧠 Model Workflow
1. Import Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

2. Load Dataset
df = pd.read_csv("diabetes.csv")
X = df.drop("Outcome", axis=1)  # Features
y = df["Outcome"]               # Target

3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

⚡ Models
🔹 Support Vector Machine (SVM)
svm_model = SVC(kernel='linear')  # try 'rbf', 'poly', 'sigmoid'
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

🔹 Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)

📊 Evaluation

Evaluate both models using accuracy, classification report, and confusion matrix.

print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))

print("\nSVM Report:\n", classification_report(y_test, y_pred_svm))
print("\nLogistic Regression Report:\n", classification_report(y_test, y_pred_lr))

print("\nSVM Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))
print("\nLogistic Regression Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))

⚙️ Hyperparameters

SVM

kernel: 'linear', 'rbf', 'poly', 'sigmoid'

C: Regularization (default=1.0)

gamma: Kernel coefficient (for 'rbf', 'poly', 'sigmoid')

Logistic Regression

solver: Algorithm for optimization ('liblinear', 'saga', 'lbfgs')

C: Inverse of regularization strength

max_iter: Maximum number of iterations (increase if convergence warning appears)

✅ Results

Both SVM and Logistic Regression are tested and compared using the diabetes dataset.
The evaluation includes:

Accuracy

Precision, Recall, F1-score (Classification Report)

Confusion Matrix
