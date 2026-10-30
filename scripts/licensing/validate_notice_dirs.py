#!/usr/bin/env python3
from pathlib import Path

from lock_notices import SKIP_NOTICE_PACKAGES, notice_dir_name, parse_uv_lock


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    lock = parse_uv_lock(repo / "uv.lock")
    root = repo / "NOTICES" / "PYPI_LICENSES"
    problems: list[str] = []
    for name, version in sorted(lock.items()):
        if name in SKIP_NOTICE_PACKAGES:
            continue
        pkg_dir = root / notice_dir_name(name, version)
        if not pkg_dir.is_dir():
            problems.append(f"missing dir: {name}=={version}")
            continue
        if not (pkg_dir / "LICENSE").exists() and not (pkg_dir / "LICENSE_NAME").exists():
            copying = pkg_dir / "NOTICES" / "COPYING"
            copying_txt = pkg_dir / "NOTICES" / "COPYING.txt"
            if copying.is_file() or copying_txt.is_file():
                continue
            problems.append(f"no LICENSE or LICENSE_NAME: {name}=={version}")
    if problems:
        print("\n".join(problems))
        raise SystemExit(1)
    print(f"OK: {len(lock) - len(SKIP_NOTICE_PACKAGES)} package notice dirs")


if __name__ == "__main__":
    main()
