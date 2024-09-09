import logging
import sys

from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)


LOG_FMT = "|%(asctime)s|%(name)s|%(levelname)s| %(message)s"


class ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": Fore.BLUE,
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.RED + Style.BRIGHT,
    }

    def format(self, record):
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
    formatter = ColoredFormatter(LOG_FMT, datefmt="%H:%M:%S")
    console_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(console_handler)

    return logger


logger = setup_logger("SGVB-PSD")
