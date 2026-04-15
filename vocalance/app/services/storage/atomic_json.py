import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


class JsonReadError(Exception):
    pass


class JsonWriteError(Exception):
    pass


def read_json_dict(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise JsonReadError(f"Missing file: {path}") from e
    except json.JSONDecodeError as e:
        raise JsonReadError(f"Invalid JSON: {path}") from e
    except OSError as e:
        raise JsonReadError(f"Cannot read: {path}") from e


def write_json_atomic(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = path.with_suffix(f".tmp.{uuid.uuid4().hex}")
    try:
        with open(staging_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        if path.exists():
            backup_path = path.with_suffix(".backup")
            os.replace(path, backup_path)
            try:
                os.replace(staging_path, path)
                if backup_path.exists():
                    os.remove(backup_path)
            except OSError:
                if backup_path.exists():
                    os.replace(backup_path, path)
                raise
        else:
            os.replace(staging_path, path)
    except OSError as e:
        if staging_path.exists():
            try:
                os.remove(staging_path)
            except OSError:
                logger.debug("Could not remove staging file %s", staging_path)
        raise JsonWriteError(f"Cannot write: {path}") from e
