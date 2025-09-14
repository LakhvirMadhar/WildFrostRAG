import os
import logging


# Define the log directory path and the log file path separately
log_dir_path = 'logging'
log_file_path = os.path.join(log_dir_path, 'app.log')

# Create the directory if it doesn't exist
os.makedirs(log_dir_path, exist_ok=True)

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=log_file_path, # Where to write to a file
    filemode='a'            # 'a' for append mode
)
