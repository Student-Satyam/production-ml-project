
from pathlib import Path
import joblib

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
housing = fetch_california_housing(as_frame=True)
X = housing.data
y = housing.target

# Prepare training data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Create the models folder if it doesn't exist
model_dir = Path("models")
model_dir.mkdir(parents=True, exist_ok=True)

# Save the trained model
model_path = model_dir / "california_housing_model.joblib"
joblib.dump(model, model_path)

print("Model saved successfully!")
print("Model path:", model_path)
print("Number of input features:", model.n_features_in_)
