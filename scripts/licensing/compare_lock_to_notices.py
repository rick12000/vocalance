#!/usr/bin/env python3
"""Compare uv.lock package versions to NOTICES/PYPI_LICENSES folders."""

from __future__ import annotations

import re
from pathlib import Path

from lock_notices import expected_notice_dirs, notice_dir_name, parse_uv_lock


def parse_notice_dirs(notices_root: Path, lock: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for child in notices_root.iterdir():
        if not child.is_dir():
            continue
        matched = False
        for name, version in sorted(lock.items(), key=lambda x: len(x[0]), reverse=True):
            if child.name == notice_dir_name(name, version):
                out[name] = version
                matched = True
                break
        if not matched:
            for name in sorted(lock, key=len, reverse=True):
                prefix = f"{name}-"
                if child.name.startswith(prefix):
                    out[name] = child.name[len(prefix) :]
                    break
            else:
                tail = child.name.rsplit("-", 1)
                if len(tail) == 2 and re.match(r"^\d", tail[1]):
                    out[tail[0]] = tail[1]
    return out


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    lock = parse_uv_lock(repo / "uv.lock")
    notices = parse_notice_dirs(repo / "NOTICES" / "PYPI_LICENSES", lock)

    lock_names = set(lock)
    notice_names = set(notices)

    print("=== In uv.lock but missing from NOTICES (any version) ===")
    for n in sorted(lock_names - notice_names):
        print(f"  {n}=={lock[n]}")

    print("\n=== In NOTICES but not in uv.lock ===")
    for n in sorted(notice_names - lock_names):
        print(f"  {n}-{notices[n]}")

    print("\n=== Version mismatch ===")
    for n in sorted(lock_names & notice_names):
        if lock[n] != notices[n]:
            print(f"  {n}: lock={lock[n]} notices={notices[n]}")

    print("\n=== NOTICE dirs with no matching uv.lock package ===")
    exact = expected_notice_dirs(lock)
    for child in sorted((repo / "NOTICES" / "PYPI_LICENSES").iterdir()):
        if not child.is_dir():
            continue
        if child.name in exact:
            continue
        if any(child.name.startswith(f"{n}-") for n in lock):
            continue
        print(f"  {child.name}")


if __name__ == "__main__":
    main()
