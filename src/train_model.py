
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the real dataset
housing = fetch_california_housing(as_frame=True)

X = housing.data
y = housing.target

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create the ML model
model = LinearRegression()

# Train the model using training data
model.fit(X_train, y_train)

# Predict values for unseen testing data
predictions = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Model training completed!")
print("Number of training rows:", len(X_train))
print("Number of testing rows:", len(X_test))

print("\nModel Evaluation:")
print("MAE:", round(mae, 4))
print("MSE:", round(mse, 4))
print("RMSE:", round(mse ** 0.5, 4))
print("R2 Score:", round(r2, 4))
