
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Create the FastAPI application
app = FastAPI(title="California Housing Prediction API")

# Find the project root and saved model
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "california_housing_model.joblib"

# Load the trained model when the application starts
model = joblib.load(MODEL_PATH)


# Define the eight required input features
class HousingInput(BaseModel):
    MedInc: float = Field(gt=0, le=20)
    HouseAge: float = Field(gt=0, le=100)
    AveRooms: float = Field(gt=0, le=100)
    AveBedrms: float = Field(gt=0, le=25)
    Population: float = Field(gt=0, le=100000)
    AveOccup: float = Field(gt=0, le=1000)
    Latitude: float = Field(ge=32, le=42)
    Longitude: float = Field(ge=-125, le=-114)


# Home endpoint
@app.get("/")
def home():
    return {"message": "California Housing Prediction API is running"}


# Health endpoint
@app.get("/health")
def health():
    return {"status": "healthy"}


# Prediction endpoint
@app.post("/predict")
def predict(data: HousingInput):
    try:
        # Convert the input into a DataFrame with named columns
        features = pd.DataFrame([data.model_dump()])

        # Generate a prediction
        prediction = float(model.predict(features)[0])

        return {
            "predicted_median_house_value_units": round(prediction, 4),
            "predicted_median_house_value_dollars": round(prediction * 100000, 2)
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        ) from error
