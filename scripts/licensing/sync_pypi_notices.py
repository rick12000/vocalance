#!/usr/bin/env python3
"""Align NOTICES/PYPI_LICENSES with uv.lock without wiping manual license trees."""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
from pathlib import Path

from fetch_licenses import LicenseFetcher, Package
from lock_notices import (
    LEGACY_NOTICE_DIR_PREFIXES,
    SKIP_NOTICE_PACKAGES,
    notice_dir_name,
    parse_uv_lock,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def target_dir(output_dir: Path, package: Package) -> Path:
    return output_dir / notice_dir_name(package.name, package.version)


def find_existing_dirs(output_dir: Path, package: Package) -> list[Path]:
    target_name = notice_dir_name(package.name, package.version)
    prefixes = (package.name, *LEGACY_NOTICE_DIR_PREFIXES.get(package.name, ()))
    matches: list[Path] = []
    seen: set[str] = set()
    for prefix in prefixes:
        pattern = re.compile(rf"^{re.escape(prefix)}-(.+)$")
        for child in output_dir.iterdir():
            if not child.is_dir() or child.name == target_name:
                continue
            if pattern.match(child.name) and child.name not in seen:
                matches.append(child)
                seen.add(child.name)
    return sorted(matches, key=lambda p: p.name)


def lock_package_names(packages: list[Package]) -> set[str]:
    names: set[str] = set()
    for pkg in packages:
        names.add(pkg.name)
        names.update(LEGACY_NOTICE_DIR_PREFIXES.get(pkg.name, ()))
    return names


def is_orphan_dir(dir_name: str, known_prefixes: set[str]) -> bool:
    for prefix in sorted(known_prefixes, key=len, reverse=True):
        if dir_name.startswith(f"{prefix}-"):
            return False
    return True


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def sync_versions(fetcher: LicenseFetcher, packages: list[Package]) -> None:
    output_dir = fetcher.output_dir
    for package in packages:
        if package.name in SKIP_NOTICE_PACKAGES:
            continue
        dest = target_dir(output_dir, package)
        if dest.exists():
            continue
        sources = find_existing_dirs(output_dir, package)
        if len(sources) == 1:
            logger.info("Preserving manual tree: %s -> %s", sources[0].name, dest.name)
            copy_tree(sources[0], dest)
            shutil.rmtree(sources[0])
        elif len(sources) > 1:
            logger.warning("Multiple notice dirs for %s; keeping newest name %s", package.name, sources[-1].name)
            copy_tree(sources[-1], dest)
            for src in sources:
                if src.exists():
                    shutil.rmtree(src)


def remove_orphans(fetcher: LicenseFetcher, packages: list[Package]) -> None:
    known = lock_package_names(packages)
    for child in list(fetcher.output_dir.iterdir()):
        if not child.is_dir():
            continue
        if is_orphan_dir(child.name, known):
            logger.info("Removing orphan notice dir: %s", child.name)
            shutil.rmtree(child)


def fetch_missing_or_refresh(
    fetcher: LicenseFetcher,
    packages: list[Package],
    refresh_names: frozenset[str],
) -> tuple[list[str], list[str]]:
    missing: list[str] = []
    metadata_only: list[str] = []
    for package in packages:
        if package.name in SKIP_NOTICE_PACKAGES:
            continue
        dest = target_dir(fetcher.output_dir, package)
        needs_fetch = not dest.exists() or package.name in refresh_names
        if not needs_fetch:
            continue
        if dest.exists():
            logger.info("Refreshing license files from PyPI: %s==%s", package.name, package.version)
            shutil.rmtree(dest)
        else:
            logger.info("Fetching license files from PyPI: %s==%s", package.name, package.version)
        status, _ = fetcher.process_package(package)
        if status == "missing":
            missing.append(f"{package.name}=={package.version}")
        elif status == "metadata_only":
            metadata_only.append(f"{package.name}=={package.version}")
    return missing, metadata_only


def cleanup_stale_version_dirs(output_dir: Path, lock: dict[str, str]) -> None:
    expected = {notice_dir_name(name, version) for name, version in lock.items()}
    for child in list(output_dir.iterdir()):
        if child.is_dir() and child.name not in expected:
            logger.info("Removing stale notice dir: %s", child.name)
            shutil.rmtree(child)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync PYPI license folders with uv.lock")
    parser.add_argument("--dry-run", action="store_true", help="Report actions only")
    parser.add_argument(
        "--refresh",
        nargs="*",
        default=(),
        metavar="PACKAGE",
        help="Re-fetch license files from PyPI for these distribution names (overwrites that package folder)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    uv_lock = repo_root / "uv.lock"
    fetcher = LicenseFetcher(str(uv_lock))
    fetcher.parse_uv_lock()
    packages = [p for p in fetcher.packages if p.name not in SKIP_NOTICE_PACKAGES]
    refresh_names = frozenset(args.refresh)

    if args.dry_run:
        logger.info("Dry run: %s packages in lock", len(packages))
        return 0

    fetcher.output_dir.mkdir(parents=True, exist_ok=True)
    lock_map = parse_uv_lock(uv_lock)
    remove_orphans(fetcher, packages)
    sync_versions(fetcher, packages)
    missing, metadata_only = fetch_missing_or_refresh(fetcher, packages, refresh_names)
    cleanup_stale_version_dirs(fetcher.output_dir, lock_map)

    logger.info("Sync complete. missing=%s metadata_only=%s", len(missing), len(metadata_only))
    if missing:
        logger.warning("Packages needing manual license follow-up:")
        for item in sorted(missing):
            logger.warning("  - %s", item)
    if metadata_only:
        logger.warning("Packages with PyPI metadata-only license (review if manual text preferred):")
        for item in sorted(metadata_only):
            logger.warning("  - %s", item)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
