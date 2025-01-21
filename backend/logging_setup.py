import logging
import logging.config
import os

from rich.console import Console
from rich.logging import RichHandler

console = Console(width=150)
rich_handler = RichHandler(console=console)

class AccessLogFilter(logging.Filter):
    """Custom logging filter to exclude certain endpoints from access logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter out log records for specified endpoints when log level is INFO or below.

        Args:
            record (logging.LogRecord): The log record to be filtered.

        Returns:
            bool: True if the log record should be processed, False otherwise.
        """

        # Define the list of endpoints to exclude
        excluded_endpoints = ["/", "/favicon.ico", "/openapi.json"]
        
        # Check if the log level is INFO or below
        if record.levelno <= logging.INFO:
            # Ensure record.args exists and is a tuple with at least 3 elements
            if hasattr(record, "args") and isinstance(record.args, tuple) and len(record.args) >= 3:
                # Exclude logs for specified endpoints
                return record.args[2] not in excluded_endpoints
        # Allow logs for other levels
        return True


# Create centralized logging configuration
def setup_logging():
    logging_config = {
        "version": 1,
        "disable_existing_loggers": True,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(name)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
                # "class": "pythonjsonlogger.jsonlogger.JsonFormatter",
            },
        },
        "handlers": {
            "console": {
                "()": lambda: rich_handler,
                "formatter": "standard",
                "level": "INFO",
                "filters": [AccessLogFilter()],
            }
        },
        "loggers": {
            "": {  # root logger
                "handlers": ["console"],
                "level": "INFO",
                "propagate": False,
            },
            "uvicorn.error": {"handlers": ["console"], "level": "WARNING", "propagate": False},
            "uvicorn.access": {"handlers": ["console"], "level": "INFO", "propagate": False},
        },
    }

    logging.config.dictConfig(logging_config)


def get_logger(name):
    return logging.getLogger(name)


setup_logging()