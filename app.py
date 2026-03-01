import streamlit as st
import pandas as pd
import pickle

# Load model
with open("churn_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
st.title("📊 Customer Churn Prediction App")

# Sidebar inputs
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
MonthlyCharges = st.sidebar.number_input("Monthly Charges", 0.0, 200.0, 70.0)
TotalCharges = st.sidebar.number_input("Total Charges", 0.0, 10000.0, 1000.0)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
Partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
Dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
Contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaymentMethod = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

if st.button("🔍 Predict Churn"):
    input_df = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "Contract": Contract,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.progress(int(probability * 100))  # shows probability as progress bar

    if prediction == 1:
        st.error(f"⚠️ High Risk of Churn ({probability:.2%})")
    else:
        st.success(f"✅ Low Risk of Churn ({probability:.2%})")