
from fastapi import FastAPI
from src.model import predict_price

# Create FastAPI application
app = FastAPI()


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "ML API is running"
    }


# Health endpoint
@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# ML prediction endpoint
@app.get("/predict")
def predict(size: float):
    price = predict_price(size)

    return {
        "house_size": size,
        "predicted_price": round(float(price), 2)
    }
