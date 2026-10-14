from fastapi import FastAPI


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