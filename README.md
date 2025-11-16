# 🏦 Bank Loan Approval Predictor  
An ML web application that estimates the probability if a loan will be approved or not based on applicant details.  
🚀 Try out the live app [here](https://mybankloanapproval.streamlit.app/)

---

## 🚀 Overview  
This web app allows users to enter applicant features (e.g., income, loan amount, credit history) and get an instant prediction: **Approved** or **Not Approved**, along with the probability of approval.  
It is built using **Python**, **Pandas**, **scikit-learn**, and **Streamlit**, and demonstrates an end-to-end ML deployment workflow.

---

## 🧠 Model & Pipeline  
- **Algorithm:** Logistic Regression (binary classification)  
- **Feature Pipeline:**  
  1. Imputation of missing values  
  2. One-Hot Encoding for categorical variables  
  3. Standardization of numeric variables  
  4. Model training & serialization  
- **Target Variable:** `Loan_Status` (Approved = 1, Not Approved = 0)  
- **Threshold for Decision:** 0.50  
- After training the model is saved (`models/loan_model.joblib`) and loaded by the web app for inference.

---

## 📊 Inputs  
| Feature            | Type        | Description                          |
|--------------------|-------------|--------------------------------------|
| `Gender`           | Categorical | “Male” or “Female”                   |
| `Married`          | Categorical | “Yes” or “No”                        |
| `ApplicantIncome`  | Numeric     | Monthly income of the applicant ($) |
| `LoanAmount`       | Numeric     | Requested loan amount ($)           |
| `Credit_History`   | Binary      | 1 = Good history, 0 = Limited/no history |

> 💡 If your training pipeline used additional fields (e.g., `Property_Area`), include them here as well.

---

## 🧪 How to Run Locally  
```bash
# 1. (Optional) Activate virtual environment  
source .venv/bin/activate

# 2. Install dependencies  
pip install -r requirements.txt

# 3. Train/re-train the model  
python train_model.py

# 4. Launch the Streamlit app  
streamlit run app.py
