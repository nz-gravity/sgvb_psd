import logging
import sys
import time

import tensorflow as tf
from colorama import Fore, Style, init


def log_if_gpu_or_cpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        logger.info(f"GPU found: {gpus}")
    else:
        logger.info("No GPU detected. Running on CPU.")


# Initialize colorama
init(autoreset=True)


class RelativeSecondsColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": Fore.BLUE,
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.RED + Style.BRIGHT,
    }

    def __init__(self, fmt=None, datefmt=None, style="%"):
        super().__init__(fmt, datefmt, style)
        self.start_time = time.time()

    def format(self, record):
        # Calculate relative seconds
        relative_seconds = int(record.created - self.start_time)
        record.relativeSeconds = relative_seconds

        # Apply color formatting
        log_color = self.COLORS.get(record.levelname, "")
        message = super().format(record)
        return f"{log_color}{message}{Style.RESET_ALL}"


def setup_logger(name):
    # Create a custom logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Set the default logging level

    # Create a console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    # Define the log message format

    # log_format = "%(asctime)s|%(name)s|%(levelname)s| %(relativeSeconds)ds |%(message)s"
    log_format = "%(asctime)s|%(name)s|%(levelname)s| %(message)s"

    formatter = RelativeSecondsColoredFormatter(log_format, datefmt="%H:%M:%S")
    console_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(console_handler)

    return logger


logger = setup_logger("SGVB-PSD")
log_if_gpu_or_cpu()
