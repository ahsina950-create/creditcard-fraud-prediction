import streamlit as st
import pandas as pd
import joblib
from src.features import preprocess

# Load model and expected feature columns
model, expected_cols = joblib.load('models/fraud_model.pkl')

# Streamlit UI
st.title("💳 Credit Card Fraud Detection")
amount = st.number_input("Transaction Amount", min_value=0.0, value=50.0)
time = st.number_input("Transaction Time (seconds)", min_value=0.0, value=50000.0)

# Construct input row with all expected features
input_data = {col: 0 for col in expected_cols}
input_data["Amount"] = amount
input_data["Time"] = time

df = pd.DataFrame([input_data])
df = preprocess(df)

# Reorder columns to match training set
df = df[expected_cols]

if st.button("Predict Fraud"):
    pred = model.predict(df)
    if pred[0] == 1:
        st.error("Fraudulent Transaction")
    else:
        st.success("Legitimate Transaction")