import logging


class Logger:
    # ANSI escape codes for colors
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

    log_formatter = logging.Formatter(
        f"%(asctime)s - [%(levelname)s] -  %(name)s - (%(filename)s).%(funcName)s(line %(lineno)d) - %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
    )

    @staticmethod
    def get_stream_handler() -> logging.StreamHandler:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(Logger.ColoredFormatter("%(levelname)s: %(message)s"))
        return stream_handler

    @staticmethod
    def get_logger(name) -> logging.Logger:
        logger = logging.getLogger(name)

        # Check if logger already has handlers to avoid duplicate logging
        if not logger.handlers:
            logger.setLevel(logging.DEBUG)
            logger.addHandler(Logger.get_stream_handler())

        return logger
