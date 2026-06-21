from pathlib import Path
import pandas as pd
import joblib
import streamlit as st

BASE_DIR   = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"

model      = joblib.load(MODELS_DIR / "model.pkl")
le_target  = joblib.load(MODELS_DIR / "le_target.pkl")
categories = joblib.load(MODELS_DIR / "categories.pkl")

st.title("🚗 BMW Car Sales Prediction App")
st.write("🔍 Application for predicting BMW car sales success")

bmw_model    = st.selectbox("Model:",        categories['Model'])
region       = st.selectbox("Region:",       categories['Region'])
fuel_type    = st.selectbox("Gorivo (Fuel Type):", categories['Fuel_Type'])
transmission = st.selectbox("Menjac (Transmission):", categories['Transmission'])
year         = st.slider("Godina (Year):", min_value=2000, max_value=2025, value=2020)
engine_size  = st.slider("Zapremina motora (L):", min_value=1.0, max_value=6.0, value=2.0, step=0.1)
mileage       = st.number_input("Kilometraza (KM):", min_value=0, max_value=500000, value=50000, step=1000)
price         = st.number_input("Cena (USD):", min_value=5000, max_value=200000, value=40000, step=1000)
sales_volume  = st.number_input("Obim prodaje (Sales Volume):", min_value=0, max_value=9999, value=5000, step=100)

if st.button("Predict"):
    input_data = pd.DataFrame([{
        "Model":         bmw_model,
        "Year":          year,
        "Region":        region,
        "Fuel_Type":     fuel_type,
        "Transmission":  transmission,
        "Engine_Size_L": engine_size,
        "Mileage_KM":    mileage,
        "Price_USD":     price,
        "Sales_Volume":  sales_volume * 0.1,
    }])

    probabilities = model.predict_proba(input_data)[0]
    high_index    = list(le_target.classes_).index("High")
    threshold     = 0.35
    prediction    = high_index if probabilities[high_index] >= threshold else 1 - high_index
    label         = le_target.inverse_transform([prediction])[0]

    class_names = {
        "High": "High - Uspesna prodaja!",
        "Low":  "Low - Slaba prodaja."
    }

    st.subheader("Results")
    st.write("Prediction:", label)
    st.write("Explanation:", class_names[label])
