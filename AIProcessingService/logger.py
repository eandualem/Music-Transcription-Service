import sys
import logging
import traceback
from pathlib import Path


class Logger:
    # Adjusted ANSI escape codes for colors based on user's description
    COLORS = {
        "WARNING": "\033[93m",  # Yellow
        "INFO": "\033[94m",  # Blue
        "DEBUG": "\033[92m",  # Green
        "CRITICAL": "\033[91m",  # Red
        "ERROR": "\033[91m",  # Red
        "ENDC": "\033[0m",  # End color
    }

    class ColoredFormatter(logging.Formatter):
        """Custom log formatter to colorize log messages based on their level."""

        def format(self, record):
            log_message = super().format(record)
            color = Logger.COLORS.get(record.levelname, Logger.COLORS["ENDC"])
            return f"{color}{log_message}{Logger.COLORS['ENDC']}"

    console_formatter = ColoredFormatter("%(levelname)s: %(message)s")
    file_formatter = logging.Formatter(
        f"%(asctime)s - [%(levelname)s] -  %(name)s - (%(filename)s).%(funcName)s(line %(lineno)d) - %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
    )

    @staticmethod
    def get_file_handler() -> logging.FileHandler:
        # create logs folder if it doesn't exist
        log_path = Path("logs")  # Simplified path for demonstration purposes
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path / "app.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(Logger.file_formatter)
        return file_handler

    @staticmethod
    def get_stream_handler() -> logging.StreamHandler:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(Logger.console_formatter)
        return stream_handler

    @staticmethod
    def get_logger(name) -> logging.Logger:
        logger = logging.getLogger(name)

        # Check if logger already has handlers to avoid duplicate logging
        if not logger.handlers:
            logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all messages
            logger.addHandler(Logger.get_file_handler())
            logger.addHandler(Logger.get_stream_handler())

        return logger

    @staticmethod
    def set_log_level(level: str):
        """Sets the log level for the logger.

        Args:
            level (str): The desired log level (e.g., "INFO", "DEBUG").
        """
        logging.getLogger().setLevel(level)

    def error(self, msg, *args, **kwargs):
        if sys.exc_info()[1]:  # Check if there's an exception currently being handled
            exc_type, exc_value, exc_traceback = sys.exc_info()
            stack_trace = traceback.extract_tb(exc_traceback)

            # Extracting detailed error information
            curr_filename, curr_line, curr_classname, curr_method = stack_trace[-1]
            location = f"Occurred in {curr_classname} method {curr_method} of {curr_filename} (line {curr_line})."

            # Extracting caller information
            if len(stack_trace) > 1:
                prev_filename, prev_line, prev_classname, prev_method = stack_trace[-2]
                caller = f"Called by {prev_classname} method {prev_method} of {prev_filename} (line {prev_line})."
            else:
                caller = "No caller information available."

            detailed_msg = f"Error: {exc_value}\n{location}\n{caller}\nOriginal Message: {msg}"
            super().error(detailed_msg, *args, **kwargs)
        else:
            super().error(msg, *args, **kwargs)
