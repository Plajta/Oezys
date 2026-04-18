import logging
import sys
from pathlib import Path

# Main logger setup
logger = logging.getLogger("AppLogger")
logger.setLevel(logging.DEBUG)

class CustomColorFormatter(logging.Formatter):
    """Custom formatter providing ANSI colors for console output."""
    BLUE = "\x1b[34;20m"
    GREEN = "\x1b[32;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"

    # [TIME] | [LEVEL] | [TAG] Message
    raw_format = "%(asctime)s | %(levelname)-7s | [%(custom_tag)s] %(message)s"

    FORMATS = {
        logging.DEBUG: BLUE + raw_format + RESET,
        logging.INFO: GREEN + raw_format + RESET,
        logging.WARNING: YELLOW + raw_format + RESET,
        logging.ERROR: RED + raw_format + RESET,
        logging.CRITICAL: BOLD_RED + raw_format + RESET
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.raw_format)
        
        # Ensure custom_tag is always present even if we somehow call logger directly
        if not hasattr(record, 'custom_tag'):
            record.custom_tag = "App"
            
        formatter = logging.Formatter(log_fmt, datefmt="%H:%M:%S")
        return formatter.format(record)

# Setup Console Handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(CustomColorFormatter())
logger.addHandler(console_handler)

# Optional function to easily add file logging
def enable_file_logging(log_filename: str = "app.log"):
    """
    Call this function once to also pipe all logs to a file.
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_filename)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_path)
    file_formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | [%(custom_tag)s] %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

# The C/Android style log macros you requested
def LOGI(tag: str, msg: str):
    """Log Info"""
    logger.info(msg, extra={"custom_tag": tag})

def LOGE(tag: str, msg: str):
    """Log Error"""
    logger.error(msg, extra={"custom_tag": tag})

def LOGW(tag: str, msg: str):
    """Log Warning"""
    logger.warning(msg, extra={"custom_tag": tag})

def LOGD(tag: str, msg: str):
    """Log Debug"""
    logger.debug(msg, extra={"custom_tag": tag})


if __name__ == "__main__":
    # Example usage:
    # enable_file_logging("logs/run.log")  # Uncomment to test file logging
    LOGI("Main", "System booted up successfully!")
    LOGW("Network", "Response delay is unusually high.")
    LOGE("Database", "Connection timeout. Failed to fetch data.")
    LOGD("Parser", "Parsing config.conf completed.")
