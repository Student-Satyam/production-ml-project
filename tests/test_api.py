
from fastapi.testclient import TestClient

from src.main import app


# Create test client
client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "ML API is running"
    }


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy"
    }


# Test ML prediction endpoint
def test_prediction():
    response = client.get("/predict?size=2200")

    assert response.status_code == 200

    data = response.json()

    assert data["house_size"] == 2200.0
    assert data["predicted_price"] == 110.0
