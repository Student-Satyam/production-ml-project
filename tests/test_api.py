
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert "API is running" in response.json()["message"]


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_prediction():
    payload = {
        "MedInc": 5.0,
        "HouseAge": 20.0,
        "AveRooms": 5.0,
        "AveBedrms": 1.0,
        "Population": 1000.0,
        "AveOccup": 3.0,
        "Latitude": 34.0,
        "Longitude": -118.0
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["predicted_median_house_value_units"] == 2.4731
    assert data["predicted_median_house_value_dollars"] == 247310.87


def test_prediction_rejects_missing_features():
    response = client.post("/predict", json={"MedInc": 5.0})

    assert response.status_code == 422


def test_prediction_rejects_invalid_latitude():
    payload = {
        "MedInc": 5.0,
        "HouseAge": 20.0,
        "AveRooms": 5.0,
        "AveBedrms": 1.0,
        "Population": 1000.0,
        "AveOccup": 3.0,
        "Latitude": 90.0,
        "Longitude": -118.0
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422



def test_metrics_endpoint():
    response = client.get("/metrics")

    assert response.status_code == 200

    data = response.json()

    assert "prediction_requests_total" in data
    assert "prediction_success_total" in data
    assert "prediction_failures_total" in data
    assert "prediction_latency_average_seconds" in data