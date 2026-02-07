import logging
import sys
import os

# Detect debugging
# Option 1: Python __debug__ is True normally, False with -O flag
# Option 2: Use environment variable DEBUG=True for dev/debug
DEBUG = False #os.getenv("DEBUG", "False").lower() in ("1", "true", "yes") or __debug__

class DebugLogger(logging.Logger):
    """
    A logger that only prints messages when debugging.
    """
    def __init__(self, name: str):
        super().__init__(name)
        if not self.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            handler.setFormatter(formatter)
            self.addHandler(handler)
            self.setLevel(logging.INFO)

        # Disable output if not debugging
        if not DEBUG:
            self.disabled = True

def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger object. Prints messages only in debug mode.
    """
    logging.setLoggerClass(DebugLogger)
    logger = logging.getLogger(name)
    return logger
