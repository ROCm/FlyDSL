import logging
import sys

__all__ = ["log"]

_FORMAT = "%(asctime)s - %(levelname)-8s - [%(filename)s : %(funcName)s] - %(message)s"
_FORMAT_SIMPLE = "| %(levelname)-8s - [%(funcName)s] - %(message)s"

_logger: logging.Logger = None
_initialized = False


def _init_logger():
    global _logger, _initialized
    if _initialized:
        return

    from .env import debug

    _logger = logging.getLogger("flydsl")
    _logger.setLevel(logging.DEBUG)
    _logger.propagate = False

    level = getattr(logging, debug.log_level)

    if debug.log_to_console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(logging.Formatter(_FORMAT_SIMPLE))
        console_handler.setLevel(level)
        _logger.addHandler(console_handler)

    if debug.log_to_file:
        file_handler = logging.FileHandler(debug.log_to_file, mode="w", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_FORMAT))
        file_handler.setLevel(level)
        _logger.addHandler(file_handler)

    if not _logger.handlers:
        _logger.addHandler(logging.NullHandler())

    _initialized = True


def log() -> logging.Logger:
    if not _initialized:
        _init_logger()
    return _logger
