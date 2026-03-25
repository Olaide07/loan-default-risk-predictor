import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# =========================

# LOAD MODEL

# =========================

with open("xgb_model.pkl", "rb") as f:
	model = pickle.load(f)

# =========================

# PAGE CONFIG

# =========================

st.set_page_config(page_title="Loan Default Risk Predictor", layout="centered")

st.title("💳 Loan Default Risk Predictor")
st.write("Predict the likelihood of loan default using machine learning.")

# =========================

# USER INPUTS

# =========================

st.subheader("📥 Enter Customer Details")

loan_amnt = st.number_input("Loan Amount", min_value=0.0, value=10000.0)
term = st.selectbox("Loan Term (months)", [36, 60])
int_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=12.0)
emp_length = st.number_input("Employment Length (years)", min_value=0.0, value=5.0)
annual_inc = st.number_input("Annual Income", min_value=0.0, value=50000.0)
dti = st.number_input("Debt-to-Income Ratio", min_value=0.0, value=15.0)
open_acc = st.number_input("Open Accounts", min_value=0, value=10)
total_acc = st.number_input("Total Accounts", min_value=0, value=20)
revol_util = st.number_input("Revolving Utilization (%)", min_value=0.0, value=50.0)

# Threshold slider

threshold = st.slider("Decision Threshold", 0.1, 0.9, 0.4)

# =========================

# PREPROCESS INPUT

# =========================

def preprocess_input():
    data = {
        "loan_amnt": loan_amnt,
        "term": term,
        "int_rate": int_rate,
        "emp_length": emp_length,
        "annual_inc": annual_inc,
        "dti": dti,
        "open_acc": open_acc,
        "total_acc": total_acc,
        "revol_util": revol_util
    }


    df = pd.DataFrame([data])

# Ensure all model features exist
    for col in model.feature_names_in_:
        if col not in df.columns:
            df[col] = 0

# Reorder columns
    df = df[model.feature_names_in_]

    return df


# =========================

# PREDICTION

# =========================

if st.button("🔍 Predict"):
    input_df = preprocess_input()


    prob = model.predict_proba(input_df)[0][1]

    st.subheader("📊 Prediction Result")

    if prob > threshold:
        st.error(f"⚠️ High Risk of Default ({prob:.2%})")
    else:
        st.success(f"✅ Low Risk ({prob:.2%})")


# =========================

# FEATURE IMPORTANCE

# =========================

st.subheader("📈 Model Insights")

if st.checkbox("Show Feature Importance"):
    importance = model.feature_importances_
    features = model.feature_names_in_


    feat_imp = pd.Series(importance, index=features).sort_values(ascending=False)

    fig, ax = plt.subplots()
    feat_imp.head(15).plot(kind="barh", ax=ax)
    ax.set_title("Top 15 Important Features")
    ax.invert_yaxis()

    st.pyplot(fig)

