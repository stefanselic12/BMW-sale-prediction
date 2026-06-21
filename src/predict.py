from pathlib import Path
import pandas as pd
import joblib

BASE_DIR   = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"

model     = joblib.load(MODELS_DIR / "model.pkl")
le_target = joblib.load(MODELS_DIR / "le_target.pkl")

new_car = pd.DataFrame([{
    "Model":         "X3",
    "Year":          2020,
    "Region":        "Europe",
    "Fuel_Type":     "Diesel",
    "Transmission":  "Automatic",
    "Engine_Size_L": 2.0,
    "Mileage_KM":    45000,
    "Price_USD":     38000,
    "Sales_Volume":  800,
}])

prediction    = model.predict(new_car)
probabilities = model.predict_proba(new_car)[0]
label         = le_target.inverse_transform(prediction)[0]

print("Predikcija za vozilo:")
print(new_car.to_string(index=False))
print()
print(f"Predvidjena klasifikacija: {label}")
print("Verovatnoce:")
for cls, prob in zip(le_target.classes_, probabilities):
    print(f"  {cls}: {prob * 100:.1f}%")
