# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
import os
import plotly.graph_objects as go

MODEL_PATH = "Models/xgb_churn_model.pkl"
SCALER_PATH = "Models/scaler.pkl"
COLS_PATH = "Models/cols.json"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
with open(COLS_PATH, "r") as f:
    cols = json.load(f)

OUT_CSV = "Dataset/stream_predictions_master.csv"
HISTORICAL_CSV = "Dataset/data.csv"

st.set_page_config(page_title="Telco Churn Predictor", layout="centered", page_icon="📊")

st.title("Telecom Customer Retention Predictor")
st.caption("Enter customer details to predict churn risk. After submission you will see the churn probability and a recommended action.")
st.markdown("---")

with st.form("form"):
    st.subheader("Identification")
    customerID = st.text_input("Customer ID", value="manual_001", help="Unique identifier for the customer (e.g., 7590-VHVEG).")

    st.markdown("---")
    st.subheader("Demographics")
    gender = st.selectbox("Gender", ["Male", "Female"], help="Customer's gender.")
    senior = st.selectbox("Is the customer a Senior Citizen?", ["No", "Yes"],
                          help="Select 'Yes' if the customer is a senior citizen; otherwise 'No'.")
    partner = st.selectbox("Does the customer have a Partner/Spouse?", ["No", "Yes"],
                           help="Indicates whether the customer has a partner (spouse).")
    dependents = st.selectbox("Does the customer have Dependents (children/others)?", ["No", "Yes"],
                              help="Select 'Yes' if the customer has dependents.")

    st.markdown("---")
    st.subheader("Customer Tenure & Services")
    tenure = st.number_input("Tenure — how many months has the customer stayed?", min_value=0, max_value=200, value=12,
                             help="Number of months the customer has been with the company.")
    phone = st.selectbox("Has Phone Service?", ["Yes", "No"], help="Does the customer have a phone subscription?")
    internet = st.selectbox("Type of Internet Service", ["DSL", "Fiber optic", "No"],
                            help="Choose the customer's internet service type.")
    streaming_movies = st.selectbox("Uses Streaming Movies service?", ["Yes", "No", "No internet service"],
                                    help="Select 'Yes' if they subscribe to streaming movies.")

    st.markdown("---")
    st.subheader("Billing & Contract")
    contract = st.selectbox("Type of Contract", ["Month-to-month", "One year", "Two year"],
                            help="Contract length; month-to-month customers tend to churn more.")
    paperless = st.selectbox("Is Paperless Billing enabled?", ["No", "Yes"],
                             help="If billing is paperless, select 'Yes'.")
    payment = st.selectbox("Payment Method", [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ], help="Payment method used by the customer.")

    st.markdown("---")
    st.subheader("Charges (in ₹)")
    monthly = st.number_input("Monthly Charges (in ₹)", min_value=0.0, max_value=100000.0, value=500.0,
                              help="Customer's average monthly bill amount in Indian Rupees.")
    total = st.number_input("Total Charges to date (in ₹)", min_value=0.0, max_value=1000000.0, value=6000.0,
                            help="Total amount billed to the customer so far in Rupees.")

    st.markdown("")
    submitted = st.form_submit_button("Predict & Save")

def preprocess_input(params, cols, scaler):
    """Build a single-row DataFrame matching training columns (cols)."""
    row = pd.DataFrame(columns=cols, data=[np.zeros(len(cols))])

    row.loc[0, "Partner"] = 1 if params["Partner"] == "Yes" else 0
    row.loc[0, "Dependents"] = 1 if params["Dependents"] == "Yes" else 0
    row.loc[0, "PhoneService"] = 1 if params["PhoneService"] == "Yes" else 0
    row.loc[0, "PaperlessBilling"] = 1 if params["PaperlessBilling"] == "Yes" else 0
    row.loc[0, "SeniorCitizen"] = 1 if params["SeniorCitizen"] == "Yes" else 0

    if "gender_Male" in row.columns:
        row.loc[0, "gender_Male"] = 1 if params["gender"] == "Male" else 0

    if params["InternetService"] == "Fiber optic" and "InternetService_Fiber optic" in row.columns:
        row.loc[0, "InternetService_Fiber optic"] = 1
    if params["InternetService"] == "No" and "InternetService_No" in row.columns:
        row.loc[0, "InternetService_No"] = 1

    if params["StreamingMovies"] == "Yes" and "StreamingMovies_Yes" in row.columns:
        row.loc[0, "StreamingMovies_Yes"] = 1
    if params["StreamingMovies"] == "No internet service" and "StreamingMovies_No internet service" in row.columns:
        row.loc[0, "StreamingMovies_No internet service"] = 1

    if params["Contract"] == "One year" and "Contract_One year" in row.columns:
        row.loc[0, "Contract_One year"] = 1
    if params["Contract"] == "Two year" and "Contract_Two year" in row.columns:
        row.loc[0, "Contract_Two year"] = 1

    pm_map = {
        "Credit card (automatic)": "PaymentMethod_Credit card (automatic)",
        "Electronic check": "PaymentMethod_Electronic check",
        "Mailed check": "PaymentMethod_Mailed check",
        "Bank transfer (automatic)": "PaymentMethod_Bank transfer (automatic)"
    }
    pm_col = pm_map.get(params["PaymentMethod"])
    if pm_col and pm_col in row.columns:
        row.loc[0, pm_col] = 1

    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    raw = pd.DataFrame([[params["tenure"], params["MonthlyCharges"], params["TotalCharges"]]], columns=num_cols)
    scaled = scaler.transform(raw)
    for i, c in enumerate(num_cols):
        if c in row.columns:
            row.loc[0, c] = scaled[0, i]

    row = row.fillna(0).astype(float)
    return row

if submitted:
    original_monthly = monthly
    original_total = total

    params = {
        "customerID": customerID,
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "InternetService": internet,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly / 83,
        "TotalCharges": total / 83
    }

    X_new = preprocess_input(params, cols, scaler)
    prob = model.predict_proba(X_new)[:, 1][0]
    pred = int(prob >= 0.5)

    st.markdown("### 🧠 Prediction Result")
    st.write("**Churn Probability:**", round(prob, 3))

    avg_hist = 0
    if os.path.exists(HISTORICAL_CSV):
        try:
            hist = pd.read_csv(HISTORICAL_CSV)
            if 'Churn' in hist.columns:
                hist['Churn_flag'] = hist['Churn'].map({'Yes':1,'No':0})
                avg_hist = float(hist['Churn_flag'].mean())
        except Exception:
            avg_hist = 0

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=float(prob),
        number={'valueformat': '.2f'},
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Churn Probability"},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': "red" if prob >= 0.5 else "green"},
            'steps': [
                {'range': [0, 0.4], 'color': "lightgreen"},
                {'range': [0.4, 0.7], 'color': "gold"},
                {'range': [0.7, 1.0], 'color': "lightcoral"}
            ]
        },
        delta={'reference': avg_hist}
    ))
    st.plotly_chart(fig_gauge, use_container_width=True)

    if prob >= 0.8:
        confidence = "🔴 Very High Risk"
    elif prob >= 0.6:
        confidence = "🟠 Moderate Risk"
    elif prob >= 0.4:
        confidence = "🟡 Low Risk"
    else:
        confidence = "🟢 Very Low Risk"

    st.write("**Confidence Level:**", confidence)

    if pred == 1:
        st.error("**Predicted Churn:** Yes 😟")
        st.markdown("""
        **Interpretation:**  
        This customer is **likely to churn (leave the service)**.  
        **Recommended actions:** offer retention incentives, investigate complaints, or assign a retention agent.
        """)
    else:
        st.success("**Predicted Churn:** No 😃")
        st.markdown("""
        **Interpretation:**  
        This customer is **likely to stay**.
        """)

    rec = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "user_id": customerID,
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "InternetService": internet,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges_Rs": original_monthly,
        "TotalCharges_Rs": original_total,
        "predicted_churn": pred,
        "churn_probability": float(prob)
    }

    rec_df = pd.DataFrame([rec])
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    if not os.path.exists(OUT_CSV):
        rec_df.to_csv(OUT_CSV, index=False)
    else:
        rec_df.to_csv(OUT_CSV, index=False, mode='a', header=False)

    st.success(f"Saved prediction row to {OUT_CSV}")

else:
    st.info("Fill the form and click **Predict & Save** to see the churn probability and recommended actions.")
