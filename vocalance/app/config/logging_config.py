import logging
import os
import sys
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

LOG_FILE_NAME = "app.log"


class LoggingConfigModel(BaseModel):
    """Logging configuration model controlling verbosity and output destinations.

    Configures logging level, message format, and whether logging is enabled.
    When disabled, uses NullHandler for complete silence (privacy-first mode).

    Attributes:
        appdata_dir_name: Directory name under %APPDATA% used to resolve the log path.
        level: Log verbosity level - DEBUG, INFO, WARNING, ERROR, or CRITICAL.
        format: Log message format string following Python logging formatter spec.
        enable_logs: When true, log to stdout and AppData/logs; when false, no logging output (default false).
    """

    appdata_dir_name: str
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="Log message format")
    enable_logs: bool = Field(
        default=False,
        description="Enable logging to stdout and disk under AppData/logs. When false: no log output (privacy-first).",
    )


def setup_logging(config: LoggingConfigModel) -> None:
    """Setup logging infrastructure with dual console and file handlers.

    Configures Python's logging system based on the provided configuration. When
    enabled, creates a timestamped log directory under %APPDATA%/<appdata_dir_name>/logs/
    and configures both console (stdout) and file handlers. When disabled, installs
    a NullHandler for complete silence.

    Args:
        config: Logging configuration object with appdata_dir_name, enable_logs, level, and format.
    """
    if not config.enable_logs:
        logging.basicConfig(level=logging.CRITICAL + 1, handlers=[logging.NullHandler()], force=True)
        return

    if os.name == "nt":
        base = os.environ.get("APPDATA", os.path.expanduser("~"))
    else:
        base = os.path.expanduser("~")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(base, config.appdata_dir_name, "logs", timestamp)
    os.makedirs(log_dir, exist_ok=True)

    log_file_path = os.path.join(log_dir, LOG_FILE_NAME)

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file_path, encoding="utf-8"),
    ]

    logging.basicConfig(level=config.level.upper(), format=config.format, handlers=handlers, force=True)
