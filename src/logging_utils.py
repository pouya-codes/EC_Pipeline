import logging
import json
from datetime import datetime

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

# Function to log a step
def log_step(step_name, start_time, end_time, status, details):
    log_entry = {
        "step_name": step_name,
        "start_time": start_time.isoformat() + 'Z',
        "end_time": end_time.isoformat() + 'Z',
        "execution_time_seconds": (end_time - start_time).total_seconds(),
        "status": status,
        "details": details
    }
    logger.info(json.dumps(log_entry))

# Function to log overall processing
def log_overall(start_time, end_time, warnings=[], errors=[]):
    overall_log = {
        "processing_start_time": start_time.isoformat() + 'Z',
        "processing_end_time": end_time.isoformat() + 'Z',
        "total_execution_time_seconds": (end_time - start_time).total_seconds(),
        "warnings": warnings,
        "errors": errors
    }
    logger.info(json.dumps(overall_log))