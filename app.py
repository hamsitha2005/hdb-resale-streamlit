import streamlit as st
import pandas as pd
import joblib
import os

# --------------------------
# Page config
# --------------------------
st.set_page_config(page_title="HDB Resale Price Predictor", layout="centered")
st.title("🏠 HDB Resale Price Predictor")
st.caption("Estimate an HDB resale price based on flat details (Punggol, Woodlands, Tampines).")

# --------------------------
# Safe file loading (Streamlit Cloud friendly)
# --------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")
cols_path = os.path.join(BASE_DIR, "feature_columns.pkl")

model = joblib.load(model_path)
feature_columns = joblib.load(cols_path)

# --------------------------
# Month mapping (UI -> number)
# --------------------------
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH_TO_NUM = {m: i + 1 for i, m in enumerate(MONTHS)}

# --------------------------
# Form UI
# --------------------------
with st.form("predict_form"):
    st.subheader("Flat Details")

    col1, col2 = st.columns(2)

    with col1:
        town = st.selectbox(
            "Town",
            ["PUNGGOL", "WOODLANDS", "TAMPINES"],
            help="Choose the town where the flat is located."
        )

        flat_type = st.selectbox(
            "Flat Type",
            ["2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE", "MULTI-GENERATION"],
            help="Select the flat type (number of rooms)."
        )

    with col2:
        flat_model = st.selectbox(
            "Flat Model",
            ["Improved", "Model A", "Apartment", "DBSS", "Adjoined flat", "3Gen"],
            help="Choose the flat model type."
        )

        floor_area_sqm = st.number_input(
            "Floor Area (sqm)",
            min_value=20.0,
            max_value=250.0,
            value=85.0,
            step=1.0,
            help="Typical range is ~30–150 sqm."
        )

    st.subheader("Storey & Lease")

    col3, col4 = st.columns(2)

    with col3:
        storey_mid = st.number_input(
            "Storey (approx.)",
            min_value=1.0,
            max_value=60.0,
            value=11.0,
            step=1.0,
            help="Enter the storey level (e.g., 11)."
        )

    with col4:
        lease_commence_date = st.number_input(
            "Lease Commence Year",
            min_value=1960,
            max_value=2026,
            value=2004,
            step=1,
            help="Year the lease started (from HDB records)."
        )

    st.subheader("Transaction Date")

    col5, col6 = st.columns(2)

    with col5:
        transaction_year = st.slider(
            "Transaction Year",
            min_value=2017,
            max_value=2026,
            value=2017,
            step=1
        )

    with col6:
        month_label = st.selectbox(
            "Transaction Month",
            MONTHS,
            index=0,
            help="Select month (the app converts it to a number)."
        )
        transaction_month = MONTH_TO_NUM[month_label]

    # Friendly submit
    submitted = st.form_submit_button("Predict Price")

# --------------------------
# Prediction logic
# --------------------------
if submitted:
    # Basic validation (keeps it user-friendly and avoids nonsense)
    if lease_commence_date > transaction_year:
        st.error("Lease Commence Year cannot be after the Transaction Year. Please correct it.")
    else:
        # Build input dataframe (must match training features before encoding)
        input_df = pd.DataFrame([{
            "town": town,
            "flat_type": flat_type,
            "flat_model": flat_model,
            "floor_area_sqm": floor_area_sqm,
            "storey_mid": storey_mid,
            "transaction_year": transaction_year,
            "transaction_month": transaction_month,
            "lease_commence_date": lease_commence_date,
        }])

        # One-hot encode same way, then align columns exactly to training
        input_encoded = pd.get_dummies(input_df, drop_first=True)
        input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

        # Predict
        pred = float(model.predict(input_encoded)[0])

        st.success(f"Predicted Resale Price: **${pred:,.0f}**")

