
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LinearRegression

# Connect to the same MLflow database
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Select our existing experiment
mlflow.set_experiment("House Price Prediction")

# Sample training data
X = [[1000], [1500], [2000], [2500]]
y = [50, 75, 100, 125]

# Create the model
model = LinearRegression()

# Train and log the model
with mlflow.start_run() as run:

    model.fit(X, y)

    # Save the trained model in MLflow
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )

    # Get the ID of this training run
    run_id = run.info.run_id

# Build the location of the saved model
model_uri = f"runs:/{run_id}/model"

# Register the model in MLflow Model Registry
registered_model = mlflow.register_model(
    model_uri=model_uri,
    name="HousePricePredictionModel"
)

print("Model registered successfully!")
print("Model name:", registered_model.name)
print("Model version:", registered_model.version)
print("Training run ID:", run_id)
