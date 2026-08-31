import logging
import os
from datetime import datetime

## Log file name 
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

## Log directory
LOG_DIR = "logs"

try: 
    os.makedirs(LOG_DIR, exist_ok=True)
except PermissionError:
    print("Error: You do not have permission to create this directory")
except OSError as e:
    print(f"Failed to create directory due to a system error: {e}")

logs_path = os.path.join(LOG_DIR, LOG_FILE)

# Define the exact format of the log messages
LOG_FORMAT = "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s"

# Configure the global logging settings
logging.basicConfig(
    filename=logs_path,
    format=LOG_FORMAT,
    level=logging.INFO
)

# Export a customized logger instance
logger = logging.getLogger("AgenticRAG_Logger")

## Add a StreamHandler to also print logs to the console
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
logger.addHandler(console_handler)