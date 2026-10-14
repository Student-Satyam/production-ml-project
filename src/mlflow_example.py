
import mlflow

# Use a local SQLite database for tracking
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Create or select our experiment
mlflow.set_experiment("House Price Prediction")

# Start an experiment run
with mlflow.start_run():

    # Record experiment settings
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("training_rows", 4)

    # Record a sample metric (for practice only)
    mlflow.log_metric("mae", 10.5)

    print("Experiment tracked successfully!")