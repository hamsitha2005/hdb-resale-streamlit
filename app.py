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
            ["Select town", "PUNGGOL", "WOODLANDS", "TAMPINES"],
            index=0,
            help="Choose the town where the flat is located."
        )

        flat_type = st.selectbox(
            "Flat Type",
            ["Select flat type", "2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE", "MULTI-GENERATION"],
            index=0,
            help="Select the flat type (number of rooms)."
        )

    with col2:
        flat_model = st.selectbox(
            "Flat Model",
            ["Select flat model", "Improved", "Model A", "Apartment", "DBSS", "Adjoined flat", "3Gen"],
            index=0,
            help="Choose the flat model type."
        )

        # No default-looking value: start at 0 and guide user with placeholder text.
        # We'll validate later to ensure it's > 0.
        floor_area_sqm = st.number_input(
            "Floor Area (sqm)",
            min_value=0.0,
            max_value=250.0,
            value=0.0,
            step=1.0,
            placeholder="Enter area (e.g., 85)",
            help="Enter the flat’s floor area in square metres."
        )

    st.subheader("Storey & Lease")

    col3, col4 = st.columns(2)

    with col3:
        # No default: start at 0, placeholder shown, validate later.
        storey_mid = st.number_input(
            "Storey (approx.)",
            min_value=0,
            max_value=60,
            value=0,
            step=1,
            placeholder="Enter storey (e.g., 11)",
            help="Enter the storey level (e.g., 11)."
        )

    with col4:
        lease_commence_date = st.slider(
            "Lease Commence Year",
            min_value=1960,
            max_value=2026,
            value=1960,
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
            ["Select month"] + MONTHS,
            index=0,
            help="Select month (the app converts it to a number)."
        )

    submitted = st.form_submit_button("Predict Price")

# --------------------------
# Prediction logic
# --------------------------
if submitted:
    # Validate required dropdowns
    if (
        town == "Select town" or
        flat_type == "Select flat type" or
        flat_model == "Select flat model" or
        month_label == "Select month"
    ):
        st.warning("Please select Town, Flat Type, Flat Model, and Transaction Month before predicting.")

    # Validate number inputs (since we removed “default” values)
    elif floor_area_sqm <= 0:
        st.warning("Please enter a valid Floor Area (sqm).")

    elif storey_mid <= 0:
        st.warning("Please enter a valid Storey (must be 1 or higher).")

    elif lease_commence_date > transaction_year:
        st.error("Lease Commence Year cannot be after the Transaction Year. Please correct it.")

    else:
        transaction_month = MONTH_TO_NUM[month_label]

        # Build input dataframe (must match training features before encoding)
        input_df = pd.DataFrame([{
            "town": town,
            "flat_type": flat_type,
            "flat_model": flat_model,
            "floor_area_sqm": float(floor_area_sqm),
            "storey_mid": int(storey_mid),
            "transaction_year": int(transaction_year),
            "transaction_month": int(transaction_month),
            "lease_commence_date": int(lease_commence_date),
        }])

        # One-hot encode same way, then align columns exactly to training
        input_encoded = pd.get_dummies(input_df, drop_first=True)
        input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

        # Predict
        pred = float(model.predict(input_encoded)[0])
        st.success(f"Predicted Resale Price: **${pred:,.0f}**")
