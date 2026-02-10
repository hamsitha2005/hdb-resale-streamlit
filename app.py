import streamlit as st
import pandas as pd
import joblib

# Basic page setup for the Streamlit app
st.set_page_config(page_title="HDB Resale Price Predictor", layout="centered")
st.title("HDB Resale Price Predictor")

# Load the trained model and the exact feature list used during training
model = joblib.load("model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# User inputs for categorical fields
town = st.selectbox("Town", ["PUNGGOL", "WOODLANDS", "TAMPINES"])
flat_type = st.selectbox("Flat Type", ["2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE", "MULTI-GENERATION"])
flat_model = st.selectbox("Flat Model", ["Improved", "Model A", "Apartment", "DBSS", "Adjoined flat", "3Gen"])

# User inputs for numeric fields
floor_area_sqm = st.number_input("Floor Area (sqm)", min_value=20.0, max_value=200.0, value=85.0, step=1.0)
storey_mid = st.number_input("Storey (mid)", min_value=1.0, max_value=60.0, value=11.0, step=1.0)

# Date-related inputs used by the model
transaction_year = st.number_input("Transaction Year", min_value=2017, max_value=2025, value=2017, step=1)
transaction_month = st.number_input("Transaction Month", min_value=1, max_value=12, value=1, step=1)
lease_commence_date = st.number_input("Lease Commence Year", min_value=1960, max_value=2025, value=2004, step=1)

# Put all inputs into one row DataFrame (same format as training data before encoding)
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

# Keep this consistent with how the final model was trained.
# Do not add extra features here unless the model was retrained with them.

# Convert categorical columns into one-hot encoded columns
input_encoded = pd.get_dummies(input_df, drop_first=True)

# Match the training feature order:
# missing columns are filled with 0, unexpected columns are removed
input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

# Run prediction only when user clicks the button
if st.button("Predict Price"):
    pred = model.predict(input_encoded)[0]
    # Display as currency with comma formatting
    st.success(f"Predicted Resale Price: ${pred:,.0f}")
