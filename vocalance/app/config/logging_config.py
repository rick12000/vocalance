import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


def get_cache_directory() -> str:
    """Get cache directory for logs and temporary files with dev/production awareness.

    Searches up the directory tree for pyproject.toml to detect development mode,
    using a sibling 'cache' directory if found. Falls back to platform-specific
    cache locations in production: %LOCALAPPDATA% on Windows, XDG_CACHE_HOME (or
    ~/.cache) on Unix-like systems.

    Returns:
        Absolute path to cache directory, created if it doesn't exist.
    """
    current_dir = Path(__file__).resolve().parent
    max_iterations = 10

    for _ in range(max_iterations):
        if (current_dir / "pyproject.toml").exists():
            cache_dir = current_dir / "cache"
            cache_dir.mkdir(exist_ok=True)
            return str(cache_dir)

        parent = current_dir.parent
        if parent == current_dir:
            break
        current_dir = parent

    if os.name == "nt":
        base_cache = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
        cache_dir = os.path.join(base_cache, "vocalance_voice_assistant", "cache")
    else:
        base_cache = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
        cache_dir = os.path.join(base_cache, "vocalance_voice_assistant")

    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


CACHE_DIR = get_cache_directory()
LOGS_BASE_DIR = os.path.join(CACHE_DIR, "logs")


def get_log_dir_for_run() -> str:
    """Create timestamped log directory for this application run.

    Generates a subdirectory within the logs folder using the current timestamp
    in YYYYMMDD_HHMMSS format, enabling chronological organization of log files.

    Returns:
        Absolute path to timestamped log directory, created if it doesn't exist.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(LOGS_BASE_DIR, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


LOG_FILE_NAME = "app.log"


class LoggingConfigModel(BaseModel):
    """Logging configuration model controlling verbosity and output destinations.

    Configures logging level, message format, and whether logging is enabled.
    When disabled, uses NullHandler for complete silence (privacy-first mode).

    Attributes:
        level: Log verbosity level - DEBUG, INFO, WARNING, ERROR, or CRITICAL.
        format: Log message format string following Python logging formatter spec.
        enable_logs: Enable logging to console and cache directory (default True).
    """

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="Log message format")
    enable_logs: bool = Field(
        default=True,
        description="Enable logging. When true: logs go to console AND cache directory. When false: completely silent (privacy-first). Default: False",
    )


def setup_logging(config: Any) -> None:
    """Setup logging infrastructure with dual console and file handlers.

    Configures Python's logging system based on the provided configuration. When
    enabled, creates a timestamped log directory and configures both console (stdout)
    and file handlers. When disabled, installs a NullHandler for complete silence.

    Args:
        config: Logging configuration object with enable_logs, level, and format attributes.
    """
    enable_logs = getattr(config, "enable_logs", False)
    level = config.level.upper() if hasattr(config, "level") else "INFO"
    log_format = config.format if hasattr(config, "format") else "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    if not enable_logs:
        logging.basicConfig(level=logging.CRITICAL + 1, handlers=[logging.NullHandler()], force=True)
        return

    handlers = []

    console_handler = logging.StreamHandler(sys.stdout)
    handlers.append(console_handler)

    log_dir = get_log_dir_for_run()
    log_file_path = os.path.join(log_dir, LOG_FILE_NAME)
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    handlers.append(file_handler)

    logging.basicConfig(level=level, format=log_format, handlers=handlers, force=True)

    logger_instance = logging.getLogger(__name__)
    logger_instance.debug(f"Logging configured: level={level}, file={log_file_path}")
