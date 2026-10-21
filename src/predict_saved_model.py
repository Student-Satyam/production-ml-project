
import joblib
import pandas as pd
from pathlib import Path

# Locate the saved model from the project root
model_path = Path("models/california_housing_model.joblib")

# Load the trained model
model = joblib.load(model_path)

# Create one example with all 8 required input features
sample = pd.DataFrame([{
    "MedInc": 5.0,
    "HouseAge": 20.0,
    "AveRooms": 5.0,
    "AveBedrms": 1.0,
    "Population": 1000.0,
    "AveOccup": 3.0,
    "Latitude": 34.0,
    "Longitude": -118.0
}])

# Predict the median house value
prediction = model.predict(sample)[0]

print("Saved model loaded successfully!")
print("Predicted target value:", round(float(prediction), 4))
print("Predicted value in dollars:", round(float(prediction) * 100000, 2))
