import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

BASE_DIR   = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"

@st.cache_resource
def load_model():
    model       = joblib.load(MODELS_DIR / "model.pkl")
    le_dict     = joblib.load(MODELS_DIR / "le_dict.pkl")
    le_target   = joblib.load(MODELS_DIR / "le_target.pkl")
    feat_cols   = joblib.load(MODELS_DIR / "feature_columns.pkl")
    return model, le_dict, le_target, feat_cols

st.set_page_config(page_title="BMW Sales Predictor", page_icon="🚗", layout="centered")

st.title("🚗 BMW Car Sales Predictor")
st.markdown("Unesite karakteristike vozila da bi model predvideo da li će prodaja biti **High** ili **Low**.")

try:
    model, le_dict, le_target, feat_cols = load_model()
except FileNotFoundError:
    st.error("Model nije pronađen. Prvo pokrenite `src/train.py` da biste istrenirali model.")
    st.stop()

st.divider()
st.subheader("Karakteristike vozila")

col1, col2 = st.columns(2)

with col1:
    bmw_model    = st.selectbox("Model",        options=le_dict['Model'].classes_)
    region       = st.selectbox("Region",       options=le_dict['Region'].classes_)
    color        = st.selectbox("Boja (Color)", options=le_dict['Color'].classes_)
    fuel_type    = st.selectbox("Gorivo (Fuel Type)", options=le_dict['Fuel_Type'].classes_)
    transmission = st.selectbox("Menjač (Transmission)", options=le_dict['Transmission'].classes_)

with col2:
    year         = st.slider("Godina (Year)",          min_value=2000, max_value=2025, value=2020)
    engine_size  = st.slider("Zapremina motora (L)",   min_value=1.0,  max_value=6.0,  value=2.0, step=0.1)
    mileage      = st.number_input("Kilometraža (KM)", min_value=0, max_value=500000, value=50000, step=1000)
    price        = st.number_input("Cena (USD)",       min_value=5000, max_value=200000, value=40000, step=1000)

st.divider()

if st.button("Predvidi prodaju", type="primary", use_container_width=True):
    input_data = {
        'Model':        le_dict['Model'].transform([bmw_model])[0],
        'Year':         year,
        'Region':       le_dict['Region'].transform([region])[0],
        'Color':        le_dict['Color'].transform([color])[0],
        'Fuel_Type':    le_dict['Fuel_Type'].transform([fuel_type])[0],
        'Transmission': le_dict['Transmission'].transform([transmission])[0],
        'Engine_Size_L': engine_size,
        'Mileage_KM':   mileage,
        'Price_USD':    price,
    }

    input_df = pd.DataFrame([input_data])[feat_cols]

    prediction    = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    label         = le_target.inverse_transform([prediction])[0]

    st.subheader("Rezultat predikcije")

    if label == "High":
        st.success(f"**Predviđena klasifikacija: HIGH** — Visoka verovatnoća uspešne prodaje")
    else:
        st.warning(f"**Predviđena klasifikacija: LOW** — Niska verovatnoća uspešne prodaje")

    class_names = le_target.classes_
    prob_df = pd.DataFrame({
        "Klasa": class_names,
        "Verovatnoća (%)": [round(p * 100, 1) for p in probabilities]
    })
    st.dataframe(prob_df, use_container_width=True, hide_index=True)

    confidence = max(probabilities) * 100
    st.metric("Pouzdanost modela", f"{confidence:.1f}%")

st.divider()
st.caption("BMW Car Sales Prediction | Random Forest Classifier | SAUSAU Projekat")
