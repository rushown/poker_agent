"""config/logging_setup.py — structured JSON logging for production."""
from __future__ import annotations

import sys

from loguru import logger


def configure_logging(
    level: str = "INFO",
    json_logs: bool = False,
    log_file: str = "plutus.log",
) -> None:
    logger.remove()

    if log_file:
        logger.add(
            log_file,
            level=level,
            rotation="20 MB",
            retention="7 days",
            serialize=True,
        )

    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
            "<cyan>{name}</cyan> - <level>{message}</level>"
        ),
    )
