# kokoclone/core/logging_setup.py
import logging
import os

LOG_FILE = os.path.join(os.path.dirname(__file__), "..", "kokoclone.log")
_FORMATTER = logging.Formatter(
    fmt="%(asctime)s.%(msecs)03d [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def get_kokoclone_logger(name: str) -> logging.Logger:
    """
    Return a logger named 'kokoclone.<name>' backed by kokoclone.log.
    Idempotent — calling twice with the same name returns the same logger
    without adding duplicate handlers.
    """
    logger = logging.getLogger(f"kokoclone.{name}")
    if not logger.handlers:
        fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
        fh.setFormatter(_FORMATTER)
        # Absorb handler errors (e.g. disk full) so logging failures never
        # propagate and break synthesis.
        fh.handleError = lambda record: None
        logger.addHandler(fh)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False  # don't double-log to root/basicConfig
    return logger
