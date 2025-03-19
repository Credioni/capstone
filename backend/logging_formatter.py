import logging

class CustomFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages based on level"""

    grey = "\x1b[38;20m"
    green = "\x1b[38;5;2m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_type = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format_type + reset,
        logging.INFO: green + format_type + reset,
        logging.WARNING: yellow + format_type + reset,
        logging.ERROR: red + format_type + reset,
        logging.CRITICAL: bold_red + format_type + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)