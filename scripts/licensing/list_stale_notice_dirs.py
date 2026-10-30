#!/usr/bin/env python3
from pathlib import Path

from lock_notices import expected_notice_dirs, parse_uv_lock


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    lock = parse_uv_lock(repo / "uv.lock")
    root = repo / "NOTICES" / "PYPI_LICENSES"
    expected = expected_notice_dirs(lock)
    stale = sorted(c.name for c in root.iterdir() if c.is_dir() and c.name not in expected)
    for name in stale:
        print(name)


if __name__ == "__main__":
    main()
