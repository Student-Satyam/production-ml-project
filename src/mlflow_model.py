
# Import MLflow to track our experiment
import mlflow
import mlflow.sklearn

# Import tools for dataset, training, and evaluation
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# Use our existing MLflow database
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Create or select an experiment for the real dataset
mlflow.set_experiment("California Housing Prediction")

# Load the real California Housing dataset
housing = fetch_california_housing(as_frame=True)

X = housing.data
y = housing.target

# Split data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# Create the model
model = LinearRegression()

# Record this experiment in MLflow
with mlflow.start_run():

    # Train the model
    model.fit(X_train, y_train)

    # Predict using unseen test data
    predictions = model.predict(X_test)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    # Record experiment parameters
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("dataset", "California Housing")
    mlflow.log_param("training_rows", len(X_train))
    mlflow.log_param("testing_rows", len(X_test))
    mlflow.log_param("input_features", X.shape[1])

    # Record evaluation metrics
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2_score", r2)

    # Save the trained model as an MLflow artifact
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
    )

    # Display the results
    print("California Housing experiment completed!")
    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))
    print("MAE:", round(mae, 4))
    print("MSE:", round(mse, 4))
    print("RMSE:", round(rmse, 4))
    print("R2 Score:", round(r2, 4))
    print("Experiment and model recorded in MLflow!")
