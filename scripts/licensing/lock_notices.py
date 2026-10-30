#!/usr/bin/env python3
"""Shared helpers for mapping uv.lock packages to NOTICES/PYPI_LICENSES folders."""

from __future__ import annotations

from pathlib import Path

SKIP_NOTICE_PACKAGES = frozenset({"vocalance"})

LEGACY_NOTICE_DIR_PREFIXES: dict[str, tuple[str, ...]] = {
    "roman-numerals": ("roman-numerals-py",),
}


def parse_uv_lock(path: Path) -> dict[str, str]:
    packages: dict[str, str] = {}
    name: str | None = None
    version: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "[[package]]":
            if name and version:
                packages[name] = version
            name = None
            version = None
        elif stripped.startswith("name = "):
            name = stripped.split("=", 1)[1].strip().strip('"')
        elif stripped.startswith("version = "):
            version = stripped.split("=", 1)[1].strip().strip('"')
    if name and version:
        packages[name] = version
    return packages


def notice_dir_name(package_name: str, version: str) -> str:
    return f"{package_name}-{version}"


def expected_notice_dirs(lock: dict[str, str]) -> set[str]:
    return {notice_dir_name(name, version) for name, version in lock.items()}


def lock_package_name_prefixes(lock: dict[str, str]) -> set[str]:
    names: set[str] = set(lock)
    for name in lock:
        names.update(LEGACY_NOTICE_DIR_PREFIXES.get(name, ()))
    return names
