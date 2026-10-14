
# Import MLflow to track our experiment
import mlflow

# Import tools for training and evaluating the model
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Use the same database as Day 8
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Select our experiment
mlflow.set_experiment("House Price Prediction")

# Sample data: house size and price (in thousands)
X = [[1000], [1500], [2000], [2500]]
y = [50, 75, 100, 125]

# Split data into training data and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Create the ML model
model = LinearRegression()

# Start recording this experiment in MLflow
with mlflow.start_run():

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on testing data
    predictions = model.predict(X_test)

    # Calculate the real MAE
    mae = mean_absolute_error(y_test, predictions)

    # Record experiment parameters
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("training_rows", len(X_train))
    mlflow.log_param("testing_rows", len(X_test))

    # Record the calculated metric
    mlflow.log_metric("mae", mae)

    # Save the trained model in MLflow
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )

    # Show the result
    print("Model training completed!")
    print("Actual MAE:", mae)
    print("Experiment recorded in MLflow!")
