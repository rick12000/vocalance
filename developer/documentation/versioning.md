# Versioning and Release Process

## Version Number

The canonical version is defined in one place: `version` in `pyproject.toml`. The bootstrap
script (`scripts/bootstrapping/setup.ps1`) contains a mirrored variable `$VOCALANCE_VERSION`
that must always match. A CI check (`verify-setup-script-version`) enforces this on every
commit and will fail the pipeline if the two diverge.

When bumping the version, update both fields together in the same commit.

## What Triggers a Release

The CI pipeline monitors every push to `main`. On each merge, the `draft-release` job
compares the current `pyproject.toml` version against the latest published GitHub release.
If the version is higher, a draft release is created automatically. If no version bump
occurred the job is skipped silently — code can be merged to main indefinitely without
producing a release.

The draft release requires all quality gates to pass first:

- Linting and pre-commit hooks
- Trivy security scan
- Unit tests
- Logging and activity tracking default-off checks
- Setup script version match check

Once created, the draft is invisible to end users. A developer must go into the GitHub
Releases page, add release notes, and publish it manually.

## Release Artifacts

The `draft-release` CI job produces the following release assets:

**Application zip** (`vocalance-v{VERSION}.zip`):

```
vocalance/   — application package
vocalance.py — entry point
pyproject.toml
uv.lock      — fully pinned dependency lockfile
README.md
DISCLAIMER.md
NOTICES/     — third-party licence disclosures
```

**Standalone scripts** (uploaded separately, not inside the zip):

- `setup.ps1` — installer
- `cleanup.ps1` — uninstaller

**Checksum**: `vocalance-v{VERSION}.zip.sha256`

Dev-only files (tests, CI config, docs, pre-commit config, bootstrapping scripts) are excluded from the zip.

Once published, a GitHub release is **immutable** — all assets are fixed artifacts and will not change.

## How the Bootstrap Script Uses Releases

`setup.ps1` is distributed as a standalone release asset rather than bundled in the zip.
Users fetch it directly from the latest release and run it locally. The script then downloads
the application zip at a hard-coded URL derived from `$VOCALANCE_VERSION`:

```
https://github.com/rick12000/vocalance/releases/download/v{VOCALANCE_VERSION}/vocalance-v{VOCALANCE_VERSION}.zip
```

This means every copy of `setup.ps1` always installs **exactly the version it was shipped
with**, regardless of when it is run.

The application is always installed to `C:\Program Files\Vocalance\` — a fixed,
system-scoped path that requires administrator rights (UAC prompt) and is consistent across all machines.
The install directory is locked to read/execute for standard users; only administrators and SYSTEM
can modify its contents. The Start Menu shortcut is written to the same location on every install.

After extraction, dependencies are installed with `uv sync --frozen`, which requires the
`uv.lock` file to be satisfied exactly. If any dependency resolution would deviate from the
lockfile the install aborts.

## Developer Workflow Summary

1. Develop and merge features to `main` freely — no version bump required.
2. When ready to release, bump `version` in `pyproject.toml` and `$VOCALANCE_VERSION` in
   `setup.ps1` to the same value in a single PR.
3. Merge the PR. CI creates a draft GitHub release automatically.
4. Add release notes on the GitHub Releases page and publish.
