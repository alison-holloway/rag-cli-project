"""Logging configuration for RAG CLI."""

import logging

from rich.console import Console
from rich.logging import RichHandler

from .config import get_settings

# Module-level logger
_logger: logging.Logger | None = None
_console_handler: RichHandler | None = None


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Set up and configure logging for the application.

    Args:
        verbose: If True, show INFO level logs. If False, only show WARNING+.

    Returns:
        Configured logger instance.
    """
    global _logger, _console_handler

    if _logger is not None:
        return _logger

    settings = get_settings()
    log_settings = settings.logging

    # Create logger - always capture all levels internally
    logger = logging.getLogger("rag_cli")
    logger.setLevel(logging.DEBUG)

    # Clear any existing handlers
    logger.handlers.clear()

    # Determine console log level based on verbose flag
    # Default: WARNING (quiet mode) - only show warnings and errors
    # Verbose: INFO - show informational messages
    console_level = logging.INFO if verbose else logging.WARNING

    # Console handler with Rich formatting
    console_handler = RichHandler(
        console=Console(stderr=True),
        show_time=True,
        show_path=False,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
    )
    console_handler.setLevel(console_level)
    console_format = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    _console_handler = console_handler

    # File handler (if configured) - always logs at configured level
    log_path = log_settings.log_path
    if log_path:
        # Ensure log directory exists
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(getattr(logging, log_settings.log_level))
        file_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    _logger = logger
    return logger


def set_log_level(level: int) -> None:
    """Set the console logging level.

    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO, logging.WARNING)
    """
    global _console_handler

    if _console_handler is not None:
        _console_handler.setLevel(level)
        # Enable local variables in tracebacks for DEBUG level
        if level == logging.DEBUG:
            _console_handler.tracebacks_show_locals = True


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Optional name for a child logger. If None, returns the main logger.

    Returns:
        Logger instance.
    """
    main_logger = setup_logging()

    if name:
        return main_logger.getChild(name)
    return main_logger


class LoggerMixin:
    """Mixin class to add logging capability to any class."""

    @property
    def logger(self) -> logging.Logger:
        """Get a logger for this class."""
        return get_logger(self.__class__.__name__)
