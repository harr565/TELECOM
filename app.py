# app.py (Telecom churn prediction with Streamlit)

import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Optional: train a model and save it (do this once, then reuse)
def train_model():
    # Example: use Telco Customer Churn dataset
    df = pd.read_csv("telco_churn.csv")  # download from Kaggle
    df = df.dropna()  # for simple demo

    # Encode categorical columns
    le_dict = {}
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    X = df.drop(columns=["Churn"])      # assuming target is "Churn"
    y = df["Churn"]

    model = RandomForestClassifier()
    model.fit(X, y)

    # Save model and encoders
    with open("churn_model.pkl", "wb") as f:
        pickle.dump({"model": model, "le_dict": le_dict}, f)

# Uncomment once to train & save
# train_model()

# Load model and encoders
with open("churn_model.pkl", "rb") as f:
    saved = pickle.load(f)
    model = saved["model"]
    le_dict = saved["le_dict"]

st.title("📞 Telecom Churn Prediction")

# Input fields (simplified; map to your actual columns)
tenure = st.number_input("Tenure (months)", min_value=0)
monthly_charges = st.number_input("Monthly Charges (₹)", min_value=0.0)
total_charges = st.number_input("Total Charges (₹)", min_value=0.0)

# Example extra categorical: contract type
contract = st.selectbox("Contract Type", ["Month‑to‑Month", "One year", "Two year"])

# Example binary: PaperlessBilling
paperless_billing = st.selectbox("Paperless Billing?", ["Yes", "No"])

# Convert selected values to numbers (you must match your train columns exactly)
input_dict = {
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": str(total_charges),  # force to str if saved as str in train
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    # ... add other features as needed
}

# Build a DataFrame with same column order as training
input_df = pd.DataFrame([input_dict])

# Label‑encode the same way as training
for col in input_df.columns:
    if col in le_dict:
        le = le_dict[col]
        input_df[col] = le.transform(input_df[col])

# Predict
if st.button("Predict Churn"):
    proba = model.predict_proba(input_df)[0, 1]  # probability of churn
    if proba > 0.5:
        st.error(f"⚠️ High churn risk: {proba:.2%}")
    else:
        st.success(f"✅ Low churn risk: {1-proba:.2%}")
