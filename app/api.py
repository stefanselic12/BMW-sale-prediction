from pathlib import Path
import pandas as pd
import joblib

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

BASE_DIR   = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"

model     = joblib.load(MODELS_DIR / "model.pkl")
le_target = joblib.load(MODELS_DIR / "le_target.pkl")


class CarInput(BaseModel):
    Model: str
    Year: int
    Region: str
    Fuel_Type: str
    Transmission: str
    Engine_Size_L: float
    Mileage_KM: int
    Price_USD: int
    Sales_Volume: int


@app.get("/")
def home():
    return {
        "message": "BMW Car Sales Prediction API is running..."
    }


@app.post("/predict")
def predict(car: CarInput):
    input_data = pd.DataFrame([{
        "Model":         car.Model,
        "Year":          car.Year,
        "Region":        car.Region,
        "Fuel_Type":     car.Fuel_Type,
        "Transmission":  car.Transmission,
        "Engine_Size_L": car.Engine_Size_L,
        "Mileage_KM":    car.Mileage_KM,
        "Price_USD":     car.Price_USD,
        "Sales_Volume":  car.Sales_Volume * 0.1,
    }])

    prediction = model.predict(input_data)
    label      = le_target.inverse_transform(prediction)[0]

    return {
        "prediction": label
    }
