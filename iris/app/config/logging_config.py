import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


def get_cache_directory() -> str:
    """
    Get cache directory for logs and temporary files following best practices.

    In development: uses repo_root/cache
    In installed package: uses system-appropriate cache directory:
    - Windows: %LOCALAPPDATA%\\iris_voice_assistant\\cache
    - Unix: $XDG_CACHE_HOME/iris_voice_assistant or ~/.cache/iris_voice_assistant

    Cache directory should be:
    - Automatically cleaned by OS during cleanup routines
    - Never backed up or synced
    - Non-essential temporary data
    - User understands it can be deleted anytime
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
        # Use %LOCALAPPDATA% for Windows cache (best practice)
        base_cache = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
        cache_dir = os.path.join(base_cache, "iris_voice_assistant", "cache")
    else:  # Unix-like (Linux, macOS)
        # Use $XDG_CACHE_HOME or ~/.cache/ (best practice for Unix)
        base_cache = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
        cache_dir = os.path.join(base_cache, "iris_voice_assistant")

    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


CACHE_DIR = get_cache_directory()
LOGS_BASE_DIR = os.path.join(CACHE_DIR, "logs")


def get_log_dir_for_run() -> str:
    """Create timestamped log directory for this run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(LOGS_BASE_DIR, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


LOG_FILE_NAME = "app.log"


class LoggingConfigModel(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="Log message format")
    enable_logs: bool = Field(
        default=True,
        description="Enable logging. When true: logs go to console AND cache directory. When false: completely silent (privacy-first). Default: False",
    )


def setup_logging(config: Any) -> None:
    """
    Setup logging with single enable_logs parameter.

    When enable_logs=True:
    - Logs printed to console (stdout)
    - Logs saved to cache directory
    - Full diagnostic information available

    When enable_logs=False:
    - Completely silent (NullHandler)
    - No console output
    - Maximum privacy (no logs anywhere)
    """
    enable_logs = getattr(config, "enable_logs", False)
    level = config.level.upper() if hasattr(config, "level") else "INFO"
    log_format = config.format if hasattr(config, "format") else "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    if not enable_logs:
        logging.basicConfig(
            level=logging.CRITICAL + 1, handlers=[logging.NullHandler()], force=True  # Higher than CRITICAL to suppress everything
        )
        return

    handlers = []

    # Console output
    console_handler = logging.StreamHandler(sys.stdout)
    handlers.append(console_handler)

    # File output to cache directory
    log_dir = get_log_dir_for_run()
    log_file_path = os.path.join(log_dir, LOG_FILE_NAME)
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    handlers.append(file_handler)

    logging.basicConfig(level=level, format=log_format, handlers=handlers, force=True)

    # Log the configuration
    logger_instance = logging.getLogger(__name__)
    logger_instance.info(f"Logging enabled: level={level}, file={log_file_path}, cache_dir={CACHE_DIR}")
