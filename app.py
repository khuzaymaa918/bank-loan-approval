import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import json



# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Bank Loan Approval Predictor", page_icon="", layout="centered")

# ---------------- HEADER ----------------
st.markdown("<h1 style='text-align: center;'>🏦 Bank Loan Approval Predictor</h1>", unsafe_allow_html=True)
st.write(
    """
    <div style='text-align: center; font-size: 17px;'>
    This web app predicts whether a bank loan application is likely to be <b>Approved</b> or <b>Not Approved</b> 
    based on applicant details using a simple Machine Learning model.
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("---")

# ---------------- LOAD MODEL ----------------
MODEL_PATH = Path("models/loan_model.joblib")
if not MODEL_PATH.exists():
    st.error("❌ Model file not found. Please run `python train_model.py` first.")
    st.stop()

_loaded = joblib.load(MODEL_PATH)
model = _loaded["model"] if isinstance(_loaded, dict) and "model" in _loaded else _loaded

# ---------------- INPUT FORM ----------------
st.subheader("🔹 Enter Applicant Information")

Gender = st.selectbox("Gender", ["Male", "Female"])
Married = st.selectbox("Marital Status", ["Yes", "No"])
ApplicantIncome = st.number_input("Applicant Income ($ per month)", min_value=0, value=4000, step=100)
LoanAmount = st.number_input("Loan Amount ($)", min_value=0, value=150, step=10)
Credit_History = st.selectbox("Credit History", ["1 (Good)", "0 (No History)"])

# Format inputs for model
input_data = pd.DataFrame([{
    "Gender": Gender,
    "Married": Married,
    "ApplicantIncome": ApplicantIncome,
    "LoanAmount": LoanAmount,
    "Credit_History": int(Credit_History[0])  # extract 1 or 0
}])

# ---------------- PREDICTION BUTTON ----------------
st.markdown("<br>", unsafe_allow_html=True)
if st.button("🔮 Predict Loan Approval", use_container_width=True):
    try:
        proba = model.predict_proba(input_data)[0, 1]
        prediction = "Approved ✅" if proba >= 0.5 else "Not Approved ❌"

        st.success(f"**Result:** {prediction}")
        st.progress(int(proba * 100))
        st.write(f"**Approval Probability:** {proba:.2%}")

        st.markdown("<br>", unsafe_allow_html=True)
        st.info("This prediction is based on a simplified dataset and should not be used for actual financial decisions.")

    except Exception as e:
        st.error(f"Prediction failed. Details: {e}")

st.markdown("---")

# ---------------- APP DETAILS TABS ----------------
tab_overview, tab_model, tab_inputs, tab_howto, tab_privacy = st.tabs(
    ["Overview", "Model Details", "Input Info", "How to Use", "Privacy"]
)

# Try to load metrics (if saved)
metrics_path = Path("reports/metrics.json")
metrics = None
if metrics_path.exists():
    try:
        metrics = json.loads(metrics_path.read_text())
    except Exception:
        metrics = None

with tab_overview:
    st.write(
        """
        This app estimates the probability that a loan application will be approved 
        using a trained **Logistic Regression** model on example data.
        """
    )
    if metrics:
        c1, c2 = st.columns(2)
        c1.metric("Test Accuracy", f"{metrics.get('test_accuracy', 0)*100:.1f}%")
        c2.metric("F1 Score", f"{metrics.get('test_f1', 0):.2f}")
    st.caption("Performance values are based on demo data — retrain for your dataset to see new results.")

with tab_model:
    st.write(
        """
        **Model Type:** Logistic Regression  
        **Pipeline Steps:**  
        1️⃣ Imputation for missing values  
        2️⃣ One-Hot Encoding for categorical features  
        3️⃣ Scaling of numeric values  
        4️⃣ Logistic Regression classification  
        
        **Target Variable:** Loan_Status (1 = Approved, 0 = Not Approved)  
        **Output:** Approval Probability
        """
    )

with tab_inputs:
    st.write(
        """
        **Input Details**
        - `Gender`: Male or Female  
        - `Marital Status`: Yes (Married) or No (Single)  
        - `Applicant Income`: Monthly income of applicant  
        - `Loan Amount`: Amount of loan requested  
        - `Credit History`: 1 (Good Record) or 0 (Limited/No Record)
        """
    )

with tab_howto:
    st.write(
        """
        **How to Use**
        1️⃣ Enter applicant details above.  
        2️⃣ Click **Predict Loan Approval**.  
        3️⃣ The app displays the approval probability and result.  
        4️⃣ To update the model, rerun `train_model.py` locally with your dataset.
        """
    )

with tab_privacy:
    st.write(
        """
        **Privacy Notice**  
        This app processes inputs **locally in your browser**.  
        No personal or financial data is stored or shared.  
        """
    )

st.markdown("---")
st.caption("Built by Khuzayma Mushtaq | Powered by scikit-learn + Streamlit")
