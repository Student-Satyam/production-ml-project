import logging


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Create logger
logger = logging.getLogger(__name__)


# Starting message
logger.info("Starting calculator")


# Try the calculation
try:
    result = 10 / 0

    print("Result:", result)

except ZeroDivisionError as error:
    logger.error(f"Calculation failed: {error}")


# Function for testing
def add(a, b):
    return a + b