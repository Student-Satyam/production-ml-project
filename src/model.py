# Import Linear Regression
from sklearn.linear_model import LinearRegression

# Import logging
import logging

# Import tools for reading .env
from dotenv import load_dotenv
import os


# Load environment variables from .env
load_dotenv()


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# Create logger
logger = logging.getLogger(__name__)


# Read configuration
model_name = os.getenv("MODEL_NAME")
model_path = os.getenv("MODEL_PATH")


# Log configuration information
logger.info(f"Model name: {model_name}")
logger.info(f"Model path: {model_path}")


# Training data
X = [[1000], [1500], [2000], [2500]]
y = [50, 75, 100, 125]


# Create ML model
logger.info("Creating ML model")

model = LinearRegression()


# Train model
logger.info("Training ML model")

model.fit(X, y)

logger.info("Model training completed")


def predict_price(size):
    """
    Predict house price based on house size.
    """

    logger.info(f"Making prediction for house size: {size}")

    prediction = model.predict([[size]])

    logger.info(f"Predicted price: {prediction[0]}")

    return prediction[0]