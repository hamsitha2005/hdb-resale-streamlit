import streamlit as st
import pandas as pd
import joblib
import os

# Configure Streamlit page settings (title shown on browser tab, wide layout for better spacing)
st.set_page_config(page_title="HDB Resale Price Predictor", layout="wide")

# Inject custom CSS to style the app container and create a card-based layout
st.markdown("""
<style>
/* Limit the maximum content width and add top padding */
.block-container { max-width: 980px; padding-top: 2rem; }

/* Card container style used to group form sections */
.card {
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 14px;
  padding: 18px 18px 6px 18px;
  background: rgba(255,255,255,0.03);
  margin-bottom: 16px;
}

/* Small helper text style */
.small-note {
  color: rgba(255,255,255,0.65);
  font-size: 0.9rem;
  margin-top: -8px;
}
</style>
""", unsafe_allow_html=True)

# Page header text
st.title("🏠 HDB Resale Price Predictor")
st.caption("Estimate an HDB resale price based on flat details (Punggol, Woodlands, Tampines).")

# Resolve file paths relative to the current script location (works locally and on Streamlit Cloud)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")
cols_path = os.path.join(BASE_DIR, "feature_columns.pkl")

# Load the trained model and the list of feature columns used during training
model = joblib.load(model_path)
feature_columns = joblib.load(cols_path)

# Mapping between month labels shown in the UI and the numeric month used by the model
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH_TO_NUM = {m: i + 1 for i, m in enumerate(MONTHS)}

# Build a form so inputs are only processed when the user clicks the submit button
with st.form("predict_form"):
    # Section 1: Flat details
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Flat Details")
    st.markdown('<p class="small-note">Enter the flat’s key characteristics.</p>', unsafe_allow_html=True)

    # Three columns to keep inputs aligned on one row
    c1, c2, c3 = st.columns([1, 1, 1])

    with c1:
        town = st.selectbox(
            "Town",
            ["Select town", "PUNGGOL", "WOODLANDS", "TAMPINES"],
            index=0
        )

    with c2:
        flat_model = st.selectbox(
            "Flat Model",
            ["Select flat model", "Improved", "Model A", "Apartment", "DBSS", "Adjoined flat", "3Gen"],
            index=0
        )

    with c3:
        floor_area_sqm = st.number_input(
            "Floor Area (sqm)",
            min_value=0.0,
            max_value=250.0,
            value=0.0,
            step=1.0,
            help="Example: 85"
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # Section 2: Storey and lease information
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Storey")

    c4, c5 = st.columns([1, 1])

    with c4:
        storey_mid = st.number_input(
            "Storey (approx.)",
            min_value=0,
            max_value=60,
            value=0,
            step=1,
            help="Example: 11"
        )

    st.subheader("Lease")
    with c5:
        lease_commence_date = st.slider(
            "Lease Commence Year",
            min_value=1960,
            max_value=2026,
            value=1990,
            step=1
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # Section 3: Transaction date
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Transaction Date")

    c6, c7 = st.columns([1, 1])

    with c6:
        transaction_year = st.slider(
            "Transaction Year",
            min_value=2017,
            max_value=2026,
            value=2020,
            step=1
        )

    with c7:
        month_label = st.selectbox(
            "Transaction Month",
            ["Select month"] + MONTHS,
            index=0
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # Submit button triggers validation + prediction
    submitted = st.form_submit_button("Predict Price")

# Run validation and prediction only after the form is submitted
if submitted:
    # Validate required categorical selections
    if town == "Select town":
        st.error("Please select a Town.")
    elif flat_model == "Select flat model":
        st.error("Please select a Flat Model.")
    elif month_label == "Select month":
        st.error("Please select a Transaction Month.")

    # Validate numeric inputs
    elif floor_area_sqm <= 0:
        st.error("Please enter a valid Floor Area (must be more than 0).")
    elif storey_mid <= 0:
        st.error("Please enter a valid Storey (must be 1 or higher).")

    # Basic logical validation to prevent impossible dates
    elif lease_commence_date > transaction_year:
        st.error("Lease Commence Year cannot be after the Transaction Year.")

    else:
        # Convert the selected month label into a numeric month
        transaction_month = MONTH_TO_NUM[month_label]

        # Create a single-row dataframe matching the feature names used before one-hot encoding
        input_df = pd.DataFrame([{
            "town": town,
            "flat_model": flat_model,
            "floor_area_sqm": float(floor_area_sqm),
            "storey_mid": int(storey_mid),
            "transaction_year": int(transaction_year),
            "transaction_month": int(transaction_month),
            "lease_commence_date": int(lease_commence_date),
        }])

        # Apply one-hot encoding and align columns to match the training feature set
        input_encoded = pd.get_dummies(input_df, drop_first=True)
        input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

        # Generate prediction and display the result
        pred = float(model.predict(input_encoded)[0])
        st.success(f"Predicted Resale Price: **${pred:,.0f}**")
