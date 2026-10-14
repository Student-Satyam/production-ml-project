# Import Python's built-in logging module
import logging


# Configure the logging system
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# Create a logger
logger = logging.getLogger(__name__)


# Normal information
logger.info("Application started")


# Warning
logger.warning("This is a warning message")


# Error
logger.error("Something went wrong")