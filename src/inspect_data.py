
from sklearn.datasets import fetch_california_housing

# Download and load the real housing dataset
housing = fetch_california_housing(as_frame=True)

# Separate input features and target
X = housing.data
y = housing.target

# Display basic information
print("Dataset loaded successfully!")

print("\nDataset shape:")
print(X.shape)

print("\nColumn names:")
print(X.columns.tolist())

print("\nFirst 5 rows:")
print(X.head())

print("\nMissing values:")
print(X.isnull().sum())

print("\nTarget summary:")
print(y.describe())
