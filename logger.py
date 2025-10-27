import sys
from loguru import logger
import os

def setup_logging(log_path: str = "logs") -> None:
    """Set up application logging configuration."""
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # Remove default logger
    logger.remove()

    # Add console logger with colors
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )

    # Add file logger for debug information
    logger.add(
        os.path.join(log_path, "debug_{time:YYYY-MM-DD}.log"),
        rotation="1 day",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG"
    )

    # Add file logger for errors only
    logger.add(
        os.path.join(log_path, "error_{time:YYYY-MM-DD}.log"),
        rotation="1 day",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="ERROR"
    )