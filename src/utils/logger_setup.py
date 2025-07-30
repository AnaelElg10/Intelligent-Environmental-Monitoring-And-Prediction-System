"""
Logger Setup for EcoVision AI

Configures logging for the application.
"""

import sys
from loguru import logger
from typing import Dict


def setup_logging(config: Dict):
    """Setup logging configuration."""
    # Remove default logger
    logger.remove()
    
    # Add console logger
    logger.add(
        sys.stdout,
        level=config.get("level", "INFO"),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # Add file logger if specified
    if config.get("file"):
        logger.add(
            config["file"],
            level=config.get("level", "INFO"),
            rotation="10 MB",
            retention="1 month",
            compression="zip"
        )