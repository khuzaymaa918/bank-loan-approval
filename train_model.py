import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
import joblib
from pathlib import Path
import json

# create a tiny sample dataset
data = pd.DataFrame({
    "Gender": ["Male","Female","Male","Male","Female","Female","Male","Male"],
    "Married": ["Yes","No","Yes","Yes","No","Yes","No","Yes"],
    "ApplicantIncome": [5000,3000,4000,6000,2500,3500,4500,5200],
    "LoanAmount": [200,100,150,250,80,120,180,220],
    "Credit_History": [1,1,0,1,1,0,1,1],
    "Loan_Status": ["Y","N","N","Y","N","N","Y","Y"]
})

y = data["Loan_Status"].map({"Y":1,"N":0})
X = data.drop(columns=["Loan_Status"])

cat_cols = ["Gender","Married"]
num_cols = ["ApplicantIncome","LoanAmount","Credit_History"]

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scale", StandardScaler())
])

pre = ColumnTransformer([
    ("cat", cat_pipe, cat_cols),
    ("num", num_pipe, num_cols)
])

model = Pipeline([
    ("pre", pre),
    ("clf", LogisticRegression(max_iter=500))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
f1 = f1_score(y_test, pred)
print(f"Accuracy: {acc:.2f}, F1: {f1:.2f}")

Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/loan_model.joblib")
print("✅ Model saved to models/loan_model.joblib")

reports = Path("reports"); reports.mkdir(exist_ok=True)
metrics = {"cv_accuracy": None, "test_accuracy": float(acc), "test_f1": float(f1)}
(reports / "metrics.json").write_text(json.dumps(metrics, indent=2))
print("📄 Saved metrics to reports/metrics.json")