from pydantic import BaseModel, Field
import logging
import sys
import os
from typing import Any
from datetime import datetime

def find_repo_root(marker_file: str) -> str:
    # Traverse upwards to find the marker file indicating repo root
    current_dir = os.path.abspath(os.path.dirname(__file__))
    while True:
        if marker_file in os.listdir(current_dir):
            return current_dir
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            raise FileNotFoundError(f"Repository root marker '{marker_file}' not found.")
        current_dir = parent_dir

REPO_MARKER_FILE = 'pyproject.toml'
REPO_ROOT = find_repo_root(REPO_MARKER_FILE)
CACHE_DIR = os.path.join(REPO_ROOT, 'cache')
LOGS_BASE_DIR = os.path.join(CACHE_DIR, 'logs')

os.makedirs(LOGS_BASE_DIR, exist_ok=True)

def get_log_dir_for_run() -> str:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(LOGS_BASE_DIR, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

LOG_FILE_NAME = 'app.log'


class LoggingConfigModel(BaseModel):
    level: str = Field(default="DEBUG")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

def setup_logging(config: Any) -> None:
    level = config.level.upper() if hasattr(config, 'level') else "INFO"
    log_format = config.format if hasattr(config, 'format') else "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    log_dir = get_log_dir_for_run()
    log_file_path = os.path.join(log_dir, LOG_FILE_NAME)

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file_path, encoding='utf-8')
    ]

    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=handlers,
        force=True
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with level: {level}, log file: {log_file_path}")
