import streamlit as st
import pandas as pd
import joblib

# Load pipeline
model = joblib.load("restaurant_rating_model.pkl")

st.set_page_config(page_title="Restaurant Rating Predictor", page_icon="🍽", layout="centered")

st.title("🍽 Restaurant Rating Prediction")
st.markdown("Enter restaurant details below to predict the expected **Aggregate Rating**.")

# Country dictionary (name -> code)
countries = {
    'India': 162,
    'United States': 1,
    'United Kingdom': 215,
    'Australia': 14,
    'Canada': 37,
    'Brazil': 30,
    'United Arab Emirates': 178,
    'South Africa': 156,
    'New Zealand': 166,
    'Singapore': 148,
    'Malaysia': 137,
    'Germany': 81,
    'France': 82,
    'Italy': 94,
    'Spain': 229,
    'Netherlands': 109,
    'Switzerland': 206,
    'Thailand': 213,
    'China': 39,
    'Japan': 60
}

important_features = [
    'Country Code',
    'Longitude',
    'Latitude',
    'Average Cost for two',
    'Has Table booking',
    'Has Online delivery'
]

input_data = {}

col1, col2 = st.columns(2)
with col1:
    country_name = st.selectbox("Select Country", list(countries.keys()))
    input_data['Country Code'] = countries[country_name]
    input_data['Longitude'] = st.number_input("Longitude")
    input_data['Latitude'] = st.number_input("Latitude")

with col2:
    input_data['Average Cost for two'] = st.number_input("Average Cost for two")
    input_data['Has Table booking'] = st.selectbox("Has Table booking", ['Yes', 'No'])
    input_data['Has Online delivery'] = st.selectbox("Has Online delivery", ['Yes', 'No'])

if st.button("🔍 Predict Rating"):
    input_df = pd.DataFrame([input_data], columns=important_features)

    # Ensure categorical columns are strings (matching training)
    categorical_cols = ['Country Code', 'Has Table booking', 'Has Online delivery']
    for col in categorical_cols:
        input_df[col] = input_df[col].astype(str)

    prediction = model.predict(input_df)[0]
    st.success(f"⭐ Predicted Rating: **{round(prediction, 2)}**")
