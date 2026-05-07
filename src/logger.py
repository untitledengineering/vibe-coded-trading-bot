import logging
import re
from src.config import API_KEY, API_SECRET

class RedactingFormatter(logging.Formatter):
    """
    Formatter that redacts sensitive information like API keys and secrets.
    """
    def __init__(self, fmt=None, datefmt=None, style='%', patterns=None):
        super().__init__(fmt, datefmt, style)
        self._patterns = patterns or []

    def format(self, record):
        msg = super().format(record)
        for pattern in self._patterns:
            if pattern:
                msg = msg.replace(pattern, "[REDACTED]")
        return msg

def setup_logging(level=logging.INFO) -> logging.Logger:
    """
    Sets up the global logger with redaction.
    """
    logger = logging.getLogger("upstox_bot")
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = RedactingFormatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            patterns=[API_KEY, API_SECRET]
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

# Initialize the global logger
logger = setup_logging()
