
from pathlib import Path
from time import perf_counter
from threading import Lock

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, make_asgi_app


# Create the FastAPI application
app = FastAPI(title="California Housing Prediction API")


# --------------------------------------------------
# 1. Prometheus metrics
# --------------------------------------------------

prediction_requests = Counter(
    "ml_prediction_requests",
    "Total number of prediction requests"
)

prediction_successes = Counter(
    "ml_prediction_successes",
    "Total number of successful predictions"
)

prediction_failures = Counter(
    "ml_prediction_failures",
    "Total number of failed predictions"
)

prediction_latency = Histogram(
    "ml_prediction_latency_seconds",
    "Time spent processing prediction requests"
)


# --------------------------------------------------
# 2. Existing JSON monitoring data
# --------------------------------------------------

monitoring_metrics = {
    "prediction_requests_total": 0,
    "prediction_success_total": 0,
    "prediction_failures_total": 0,
    "prediction_latency_total_seconds": 0.0,
}

metrics_lock = Lock()


# --------------------------------------------------
# 3. Middleware: monitor prediction requests
# --------------------------------------------------

@app.middleware("http")
async def monitor_predictions(request: Request, call_next):

    # Monitor only POST requests to /predict
    if request.url.path != "/predict" or request.method != "POST":
        return await call_next(request)

    # Start measuring request time
    start_time = perf_counter()

    # Count request in JSON monitoring
    with metrics_lock:
        monitoring_metrics["prediction_requests_total"] += 1

    # Count request in Prometheus
    prediction_requests.inc()

    try:
        # Process the API request
        response = await call_next(request)

        # Count successful and failed HTTP responses
        if response.status_code < 400:

            with metrics_lock:
                monitoring_metrics["prediction_success_total"] += 1

            prediction_successes.inc()

        else:

            with metrics_lock:
                monitoring_metrics["prediction_failures_total"] += 1

            prediction_failures.inc()

        return response

    except Exception:
        # Count unexpected errors
        with metrics_lock:
            monitoring_metrics["prediction_failures_total"] += 1

        prediction_failures.inc()
        raise

    finally:
        # Calculate request duration
        elapsed_time = perf_counter() - start_time

        # Update JSON monitoring data
        with metrics_lock:
            monitoring_metrics[
                "prediction_latency_total_seconds"
            ] += elapsed_time

        # Update Prometheus latency histogram
        prediction_latency.observe(elapsed_time)


# --------------------------------------------------
# 4. Prometheus metrics endpoint
# --------------------------------------------------

prometheus_app = make_asgi_app()

app.mount("/prometheus-metrics", prometheus_app)


# --------------------------------------------------
# 5. Load the trained ML model
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "california_housing_model.joblib"
)

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# 6. Define input validation
# --------------------------------------------------

class HousingInput(BaseModel):

    MedInc: float = Field(gt=0, le=20)
    HouseAge: float = Field(gt=0, le=100)
    AveRooms: float = Field(gt=0, le=100)
    AveBedrms: float = Field(gt=0, le=25)
    Population: float = Field(gt=0, le=100000)
    AveOccup: float = Field(gt=0, le=1000)
    Latitude: float = Field(ge=32, le=42)
    Longitude: float = Field(ge=-125, le=-114)


# --------------------------------------------------
# 7. Home endpoint
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "California Housing Prediction API is running"
    }


# --------------------------------------------------
# 8. Health endpoint
# --------------------------------------------------

@app.get("/health")
def health():

    return {"status": "healthy"}


# --------------------------------------------------
# 9. Existing JSON monitoring endpoint
# --------------------------------------------------

@app.get("/metrics")
def get_metrics():

    # Safely copy monitoring data
    with metrics_lock:
        result = monitoring_metrics.copy()

    # Calculate average request latency
    total_requests = result["prediction_requests_total"]

    if total_requests > 0:
        average_latency = (
            result["prediction_latency_total_seconds"]
            / total_requests
        )
    else:
        average_latency = 0.0

    result["prediction_latency_average_seconds"] = round(
        average_latency, 6
    )

    result["prediction_latency_total_seconds"] = round(
        result["prediction_latency_total_seconds"], 6
    )

    return result


# --------------------------------------------------
# 10. Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(data: HousingInput):

    try:
        # Convert input into a DataFrame
        features = pd.DataFrame([data.model_dump()])

        # Generate prediction
        prediction = float(model.predict(features)[0])

        return {
            "predicted_median_house_value_units": round(
                prediction, 4
            ),
            "predicted_median_house_value_dollars": round(
                prediction * 100000, 2
            )
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        ) from error
