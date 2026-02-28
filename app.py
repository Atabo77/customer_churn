import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="wide")
st.title("📊 Customer Churn Prediction App")
st.markdown("### Predict if a telecom customer is likely to churn")

# Load model safely
try:
    with open("churn_pipeline.pkl", "rb") as f:
        model = pickle.load(f)
    st.success("✅ Model loaded successfully")
except FileNotFoundError:
    st.error("⚠️ Model file not found. Run train_model.py first!")

# User input
st.sidebar.header("Customer Input Features")
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
MonthlyCharges = st.sidebar.number_input("Monthly Charges", 0.0, 200.0, 70.0)
TotalCharges = st.sidebar.number_input("Total Charges", 0.0, 10000.0, 1000.0)

# Other categorical inputs
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
Partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
Dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
Contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaymentMethod = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

# Predict button
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

    if prediction == 1:
        st.error(f"⚠️ High Risk of Churn ({probability:.2%} probability)")
    else:
        st.success(f"✅ Low Risk of Churn ({probability:.2%} probability)")