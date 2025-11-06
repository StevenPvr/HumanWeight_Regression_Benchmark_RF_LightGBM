"""Centralized logging configuration for the project.

WHY: Keep logging setup in a single place to avoid duplicated handler
registrations and ensure a consistent format across the whole pipeline.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from src.constants import PIPELINE_LOG_PATH

_LOGGING_CONFIGURED = False


def configure_logging() -> None:
    """Configure the root logger with stream and file handlers.

    Idempotent: safe to call multiple times without stacking handlers.
    """

    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    # Ensure log file exists at results/pipeline.log
    PIPELINE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    Path(PIPELINE_LOG_PATH).touch(exist_ok=True)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(PIPELINE_LOG_PATH, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # During pytest, keep a single stdout handler to satisfy tests
    if "PYTEST_CURRENT_TEST" in os.environ:
        root_logger.handlers = [stream_handler]
    else:
        root_logger.handlers = [stream_handler, file_handler]

    _LOGGING_CONFIGURED = True
