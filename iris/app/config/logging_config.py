import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


def get_cache_directory() -> str:
    """
    Get cache directory for logs and temporary files.

    In development: uses repo_root/cache
    In installed package: uses system temp or user cache directory
    """
    # Try to find repo root (development mode)
    current_dir = Path(__file__).resolve().parent
    max_iterations = 10

    for _ in range(max_iterations):
        if (current_dir / "pyproject.toml").exists():
            # Development mode - found repo root
            cache_dir = current_dir / "cache"
            cache_dir.mkdir(exist_ok=True)
            return str(cache_dir)

        parent = current_dir.parent
        if parent == current_dir:
            break
        current_dir = parent

    # Installed package mode - use platform-appropriate cache directory
    if os.name == "nt":  # Windows
        base_cache = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
        cache_dir = os.path.join(base_cache, "iris_voice_assistant", "cache")
    else:  # Unix-like
        base_cache = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
        cache_dir = os.path.join(base_cache, "iris_voice_assistant")

    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


CACHE_DIR = get_cache_directory()
LOGS_BASE_DIR = os.path.join(CACHE_DIR, "logs")

os.makedirs(LOGS_BASE_DIR, exist_ok=True)


def get_log_dir_for_run() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(LOGS_BASE_DIR, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


LOG_FILE_NAME = "app.log"


class LoggingConfigModel(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="DEBUG")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def setup_logging(config: Any) -> None:
    level = config.level.upper() if hasattr(config, "level") else "INFO"
    log_format = config.format if hasattr(config, "format") else "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    log_dir = get_log_dir_for_run()
    log_file_path = os.path.join(log_dir, LOG_FILE_NAME)

    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_file_path, encoding="utf-8")]

    logging.basicConfig(level=level, format=log_format, handlers=handlers, force=True)
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with level: {level}, log file: {log_file_path}")
