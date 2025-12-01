#!/usr/bin/env python3
"""
Fetch and organize PyPI licenses and third-party notices for packages.
Tries sdist first, then wheel, then PyPI metadata classifier, then flags as missing.
"""

import json
import logging
import re
import shutil
import sys
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.request import urlopen

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Package:
    name: str
    version: str
    sdist_url: Optional[str] = None
    wheel_urls: List[str] = field(default_factory=list)


class LicenseFetcher:
    def __init__(self, uv_lock_path: str, max_n_deps: Optional[int] = None):
        self.uv_lock_path = Path(uv_lock_path)
        repo_root = self.uv_lock_path.parent
        self.output_dir = repo_root / "NOTICES" / "PYPI_LICENSES"
        self.packages: List[Package] = []
        self.missing_licenses: List[str] = []
        self.metadata_only_licenses: List[Tuple[str, str]] = []  # (package, license)
        self.max_n_deps = max_n_deps

        # License patterns - strict matching for main license file
        self.license_patterns = [r"^LICENSE", r"^LICENCE"]

        # Notice patterns - broad matching for acknowledgements and subdependency notices
        self.notice_patterns = [
            r"^NOTICE",
            r"^THIRD_PARTY",
            r"^COPYING",
            r"^COPYRIGHT",
            r"^ACKNOWLEDGMENT",
            r"^ACKNOWLEDGEMENT",
            r"^ATTRIBUTION",
            r"^AUTHORS",
            r"^CONTRIBUTORS",
            r"^CONTRIBUTOR",
            r"^DISCLAIMER",
            r"^LEGAL",
            r"^COMPLIANCE",
            r"^PATENT",
            r"^TRADEMARK",
            r"^GRANT",
        ]

    def parse_uv_lock(self) -> None:
        logger.info(f"Parsing {self.uv_lock_path}")
        with open(self.uv_lock_path, "r", encoding="utf-8") as f:
            lines = f.read().split("\n")

        current_package = None
        for i, line in enumerate(lines):
            stripped = line.strip()

            if stripped == "[[package]]":
                if current_package and current_package["name"] and current_package["version"]:
                    self.packages.append(
                        Package(
                            name=current_package["name"],
                            version=current_package["version"],
                            sdist_url=current_package.get("sdist_url"),
                            wheel_urls=current_package.get("wheel_urls", []),
                        )
                    )
                current_package = {"name": None, "version": None, "sdist_url": None, "wheel_urls": []}
            elif current_package is not None:
                if stripped.startswith("name = "):
                    current_package["name"] = stripped.split("name = ")[1].strip().strip('"')
                elif stripped.startswith("version = "):
                    current_package["version"] = stripped.split("version = ")[1].strip().strip('"')
                elif stripped.startswith("sdist = {"):
                    match = re.search(r'url = "([^"]+)"', stripped)
                    if not match:
                        for j in range(i, min(i + 5, len(lines))):
                            m = re.search(r'url = "([^"]+)"', lines[j])
                            if m:
                                match = m
                                break
                    if match:
                        current_package["sdist_url"] = match.group(1)
                elif stripped.startswith("wheels = ["):
                    for j in range(i, min(i + 50, len(lines))):
                        m = re.search(r'url = "([^"]+\.whl)"', lines[j])
                        if m:
                            current_package["wheel_urls"].append(m.group(1))

        if current_package and current_package["name"] and current_package["version"]:
            self.packages.append(
                Package(
                    name=current_package["name"],
                    version=current_package["version"],
                    sdist_url=current_package.get("sdist_url"),
                    wheel_urls=current_package.get("wheel_urls", []),
                )
            )

        if self.max_n_deps:
            self.packages = self.packages[: self.max_n_deps]

        logger.info(f"Found {len(self.packages)} packages" + (f" (limited to {self.max_n_deps})" if self.max_n_deps else ""))

    def setup_output_directory(self) -> None:
        if self.output_dir.exists():
            logger.info("Cleaning up old output directory")
            try:
                shutil.rmtree(self.output_dir)
            except PermissionError:
                logger.warning("Could not remove old directory (in use), will overwrite")

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_package_directory(self, package: Package) -> Path:
        pkg_dir = self.output_dir / f"{package.name}-{package.version}"
        pkg_dir.mkdir(parents=True, exist_ok=True)
        return pkg_dir

    def download_and_extract_sdist(self, package: Package) -> Optional[Path]:
        if not package.sdist_url:
            return None

        try:
            extract_dir = Path(tempfile.mkdtemp())
            with urlopen(package.sdist_url, timeout=30) as response:
                if package.sdist_url.endswith((".tar.gz", ".tgz")):
                    tar_path = extract_dir / "archive.tar.gz"
                    with open(tar_path, "wb") as f:
                        f.write(response.read())
                    with tarfile.open(tar_path, "r:gz") as tar:
                        tar.extractall(extract_dir)
                    tar_path.unlink()
                elif package.sdist_url.endswith(".zip"):
                    zip_path = extract_dir / "archive.zip"
                    with open(zip_path, "wb") as f:
                        f.write(response.read())
                    with zipfile.ZipFile(zip_path, "r") as zf:
                        zf.extractall(extract_dir)
                    zip_path.unlink()
                else:
                    return None

            return extract_dir
        except Exception as e:
            logger.debug(f"Sdist download failed for {package.name}: {e}")
            return None

    def download_and_extract_wheel(self, package: Package) -> Optional[Path]:
        if not package.wheel_urls:
            return None

        try:
            extract_dir = Path(tempfile.mkdtemp())
            with urlopen(package.wheel_urls[0], timeout=30) as response:
                wheel_path = extract_dir / "archive.whl"
                with open(wheel_path, "wb") as f:
                    f.write(response.read())
                with zipfile.ZipFile(wheel_path, "r") as zf:
                    zf.extractall(extract_dir)
                wheel_path.unlink()

            return extract_dir
        except Exception as e:
            logger.debug(f"Wheel download failed for {package.name}: {e}")
            return None

    def get_license_from_pypi_metadata(self, package: Package) -> Optional[str]:
        """Extract license from PyPI metadata (license field and classifiers)."""
        try:
            url = f"https://pypi.org/pypi/{package.name}/{package.version}/json"
            with urlopen(url, timeout=30) as response:
                data = json.loads(response.read())
                info = data.get("info", {})

                # First check the license field
                license_field = info.get("license")
                if license_field and isinstance(license_field, str) and license_field.strip():
                    return license_field.strip()

                # Then check classifiers
                for classifier in info.get("classifiers", []):
                    if classifier.startswith("License ::"):
                        # Extract license name from classifier like "License :: OSI Approved :: MIT License"
                        parts = classifier.split("::")
                        if len(parts) >= 3:
                            return parts[-1].strip()

                return None
        except Exception as e:
            logger.debug(f"Failed to fetch PyPI metadata for {package.name}: {e}")
            return None

    def find_matching_files(self, root_dir: Path, patterns: List[str]) -> List[Path]:
        files = []
        seen = set()

        for path in root_dir.rglob("*"):
            if path.is_file():
                path_lower = str(path).lower()
                if any(part in path_lower for part in [".git", "__pycache__", ".egg-info", "node_modules", ".pytest"]):
                    continue

                for pattern in patterns:
                    if re.match(pattern, path.name, re.IGNORECASE):
                        if str(path) not in seen:
                            files.append(path)
                            seen.add(str(path))
                        break

        return files

    def copy_file_safe(self, src: Path, dst: Path) -> bool:
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            return True
        except Exception as e:
            logger.error(f"Failed to copy {src.name}: {e}")
            return False

    def process_package(self, package: Package) -> Tuple[str, List[str]]:
        """
        Process a package and return (status, found_files).
        Status can be: 'found', 'metadata_only', or 'missing'
        """
        logger.info(f"Processing {package.name}=={package.version}")

        pkg_dir = self.create_package_directory(package)
        found_files = []
        notices_dir = None

        # Try sdist first, then wheel
        extract_dir = self.download_and_extract_sdist(package)
        source_type = "sdist"
        if not extract_dir:
            extract_dir = self.download_and_extract_wheel(package)
            source_type = "wheel"

        if extract_dir:
            try:
                # Find and copy license files
                license_files = self.find_matching_files(extract_dir, self.license_patterns)

                if license_files:
                    logger.info(f"Found {len(license_files)} LICENSE file(s) for {package.name} ({source_type})")
                    license_copied = False
                    for lic_file in license_files:
                        if not license_copied:
                            if self.copy_file_safe(lic_file, pkg_dir / "LICENSE"):
                                found_files.append(lic_file.name)
                                license_copied = True

                        if lic_file.name.lower() not in [f.lower() for f in found_files]:
                            if notices_dir is None:
                                notices_dir = pkg_dir / "NOTICES"
                                notices_dir.mkdir(parents=True, exist_ok=True)
                            if self.copy_file_safe(lic_file, notices_dir / lic_file.name):
                                found_files.append(lic_file.name)

                # Find and copy notice files
                notice_files = self.find_matching_files(extract_dir, self.notice_patterns)

                if notice_files:
                    logger.info(f"Found {len(notice_files)} notice file(s) for {package.name}")
                    for notice_file in notice_files:
                        if notice_file.name.lower() not in [f.lower() for f in found_files]:
                            if notices_dir is None:
                                notices_dir = pkg_dir / "NOTICES"
                                notices_dir.mkdir(parents=True, exist_ok=True)
                            if self.copy_file_safe(notice_file, notices_dir / notice_file.name):
                                found_files.append(notice_file.name)

            finally:
                if extract_dir.exists():
                    shutil.rmtree(extract_dir)

        # Check if we found a real LICENSE file
        has_license = (pkg_dir / "LICENSE").exists()

        if has_license:
            return "found", found_files

        # Try PyPI metadata as fallback
        license_from_metadata = self.get_license_from_pypi_metadata(package)

        if license_from_metadata:
            logger.warning(f"[METADATA ONLY] License found in PyPI metadata for {package.name}: {license_from_metadata}")
            # Write license name to LICENSE_NAME file
            license_name_file = pkg_dir / "LICENSE_NAME"
            try:
                with open(license_name_file, "w", encoding="utf-8") as f:
                    f.write(f"{license_from_metadata}\n")
                    f.write("\n(Note: Actual license file not found in sdist or wheel.)\n")
                    f.write("(License info extracted from PyPI metadata classifier.)\n")
                    f.write(f"(See https://pypi.org/project/{package.name}/ for more details.)\n")
                found_files.append("LICENSE_NAME")
                self.metadata_only_licenses.append(f"{package.name}=={package.version}")
                return "metadata_only", found_files
            except Exception as e:
                logger.error(f"Failed to write LICENSE_NAME for {package.name}: {e}")

        # No license found anywhere
        if len(found_files) == 0:
            logger.warning(f"[MISSING] NO LICENSE FOUND for {package.name}=={package.version}")
            self.missing_licenses.append(f"{package.name}=={package.version}")
            (pkg_dir / ".gitkeep").touch()
        elif not has_license:
            logger.warning(f"[WARNING] Found notices but NO main LICENSE for {package.name}=={package.version}")
            (pkg_dir / ".gitkeep").touch()

        return "missing" if not has_license and license_from_metadata is None else "metadata_only", found_files

    def run(self) -> None:
        logger.info("=" * 80)
        mode = f"DEV MODE - TOP {self.max_n_deps}" if self.max_n_deps else "PRODUCTION"
        logger.info(f"Starting PyPI License Fetcher [{mode}]")
        logger.info("=" * 80)

        self.parse_uv_lock()
        self.setup_output_directory()

        found_count = 0
        metadata_only_count = 0
        for i, package in enumerate(self.packages, 1):
            logger.info(f"\n[{i}/{len(self.packages)}] {package.name}=={package.version}")
            try:
                status, _ = self.process_package(package)
                if status == "found":
                    found_count += 1
                elif status == "metadata_only":
                    metadata_only_count += 1
            except Exception as e:
                logger.error(f"Error processing {package.name}: {e}")
                pkg_dir = self.create_package_directory(package)
                (pkg_dir / ".gitkeep").touch()
                self.missing_licenses.append(f"{package.name}=={package.version}")

        self.generate_summary_report(found_count, metadata_only_count)

    def generate_summary_report(self, found_count: int, metadata_only_count: int) -> None:
        logger.info("\n" + "=" * 80)
        logger.info("LICENSE FETCH SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total packages: {len(self.packages)}")
        logger.info(f"Packages with LICENSE file: {found_count}")
        logger.info(f"Packages with license from PyPI metadata only: {metadata_only_count}")
        logger.info(f"Packages without license: {len(self.missing_licenses)}")
        logger.info("=" * 80)

        if metadata_only_count > 0:
            logger.warning("\n[METADATA ONLY] PACKAGES WITH LICENSE IN METADATA CLASSIFIER:")
            logger.warning("(Actual LICENSE files not found in distributions)")
            for pkg in sorted(self.metadata_only_licenses):
                logger.warning(f"  - {pkg}")

        if self.missing_licenses:
            logger.warning("\n[MISSING] PACKAGES WITH NO LICENSE FOUND:")
            for pkg in sorted(self.missing_licenses):
                logger.warning(f"  - {pkg}")

        summary_path = self.output_dir / "LICENSE_FETCH_SUMMARY.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("LICENSE FETCH SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total packages: {len(self.packages)}\n")
            f.write(f"Packages with LICENSE file found: {found_count}\n")
            f.write(f"Packages with license from PyPI metadata classifier only: {metadata_only_count}\n")
            f.write(f"Packages without any license information: {len(self.missing_licenses)}\n")
            f.write("=" * 80 + "\n\n")

            if metadata_only_count > 0:
                f.write("[METADATA ONLY] PACKAGES WITH LICENSE IN PYPI METADATA:\n")
                f.write("Note: These packages do not include LICENSE files in their distributions.\n")
                f.write("The license was extracted from PyPI metadata classifiers.\n\n")
                for pkg in sorted(self.metadata_only_licenses):
                    f.write(f"  - {pkg}\n")
                f.write("\n")

            if self.missing_licenses:
                f.write("[MISSING] PACKAGES WITHOUT ANY LICENSE INFORMATION:\n")
                f.write("These packages need manual license identification:\n\n")
                for pkg in sorted(self.missing_licenses):
                    f.write(f"  - {pkg}\n")

        logger.info(f"\nReport: {summary_path}")


def main():
    import argparse

    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    uv_lock_path = repo_root / "uv.lock"

    parser = argparse.ArgumentParser(description="Fetch PyPI licenses and notices")
    parser.add_argument("--max-deps", type=int, default=None, help="Limit to top N packages")
    args = parser.parse_args()

    if not uv_lock_path.exists():
        logger.error(f"uv.lock not found at {uv_lock_path}")
        sys.exit(1)

    logger.info(f"Output: {repo_root}/NOTICES/PYPI_LICENSES")

    fetcher = LicenseFetcher(str(uv_lock_path), max_n_deps=args.max_deps)
    fetcher.run()


if __name__ == "__main__":
    main()
