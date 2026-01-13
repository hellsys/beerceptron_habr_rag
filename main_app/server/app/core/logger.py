import logging
import logging.config
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,

    "formatters": {
        "default": {
            "format": "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
        },
        "detailed": {
            "format": "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s"
        },
    },

    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "default",
        },
        "file_app": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": LOG_DIR / "app.log",
            "encoding": "utf-8"
        },
        "file_errors": {
            "class": "logging.FileHandler",
            "level": "ERROR",
            "formatter": "detailed",
            "filename": LOG_DIR / "errors.log",
            "encoding": "utf-8"
        },
    },

    "loggers": {
        "app": {
            "handlers": ["console", "file_app"],
            "level": "DEBUG",
            "propagate": False
        },
        "errors": {
            "handlers": ["file_errors"],
            "level": "ERROR",
            "propagate": False
        }
    }
}


def setup_logging():
    logging.config.dictConfig(LOGGING_CONFIG)