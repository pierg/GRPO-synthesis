import loguru
import sys

logger = loguru.logger

def configure_logging():
    """Configure loguru logging without extra new lines."""
    logger.remove()

    log_format = "<green>{time:HH:mm:ss.SSS}</green> | <level>{level:<8}</level> | <cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

    # File logging
    logger.add(
        "inference.log",
        rotation="1 day",
        retention="7 days",
        level="DEBUG",
        format=log_format,
        colorize=True
    )

    # Console logging (use sys.stderr for cleaner default behavior)
    logger.add(
        sys.stderr,
        level="DEBUG",
        format=log_format,
        colorize=True
    )
