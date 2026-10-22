
from time import perf_counter
from threading import Lock
from fastapi import Request

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Create the FastAPI application
app = FastAPI(title="California Housing Prediction API")

# Store API monitoring data
monitoring_metrics = {
    "prediction_requests_total": 0,
    "prediction_success_total": 0,
    "prediction_failures_total": 0,
    "prediction_latency_total_seconds": 0.0,
}

# Protect counters when requests happen at the same time
metrics_lock = Lock()

@app.middleware("http")
async def monitor_predictions(request: Request, call_next):

    # Monitor only prediction requests
    if request.url.path != "/predict" or request.method != "POST":
        return await call_next(request)

    # Start measuring time
    start_time = perf_counter()

    # Count the incoming request
    with metrics_lock:
        monitoring_metrics["prediction_requests_total"] += 1

    try:
        # Send the request to the API endpoint
        response = await call_next(request)

        # Track success or failure using HTTP status
        with metrics_lock:
            if response.status_code < 400:
                monitoring_metrics["prediction_success_total"] += 1
            else:
                monitoring_metrics["prediction_failures_total"] += 1

        return response

    except Exception:
        # Count unexpected errors
        with metrics_lock:
            monitoring_metrics["prediction_failures_total"] += 1
        raise

    finally:
        # Record the time taken
        elapsed_time = perf_counter() - start_time

        with metrics_lock:
            monitoring_metrics["prediction_latency_total_seconds"] += elapsed_time

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

@app.get("/metrics")
def get_metrics():

    # Make a safe copy of the current metrics
    with metrics_lock:
        result = monitoring_metrics.copy()

    # Calculate average prediction response time
    total_requests = result["prediction_requests_total"]

    if total_requests > 0:
        average_latency = (
            result["prediction_latency_total_seconds"]
            / total_requests
        )
    else:
        average_latency = 0.0

    # Return monitoring information
    result["prediction_latency_average_seconds"] = round(
        average_latency, 6
    )

    result["prediction_latency_total_seconds"] = round(
        result["prediction_latency_total_seconds"], 6
    )

    return result


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
