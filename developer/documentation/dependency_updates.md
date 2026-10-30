# Dependency Updates

## Dependency Groups

`pyproject.toml` defines four dependency groups:

| Group | Purpose | Install command |
|-------|---------|-----------------|
| *(base)* | Core runtime dependencies | `uv sync` |
| `llm` | Optional AI features: smart dictation and AI text editing. Requires Microsoft C++ Build Tools. | `uv sync --extra llm` |
| `dev` | Development tooling: pytest, black, flake8, sphinx, prek | `uv sync --extra dev` |
| `docs` | Documentation build tools | `uv sync --extra docs` |

`llama-cpp-python` and `huggingface-hub` belong exclusively to the `llm` group and must not appear in the base `dependencies` list.

## Process

1. **Update `pyproject.toml`**:
   - Base runtime packages go under `[project] dependencies`
   - LLM-only packages go under `[project.optional-dependencies] llm`
   - Dev/tooling packages go under `[project.optional-dependencies] dev`

2. **Regenerate `uv.lock`**:
   - Temporarily remove only the `dev` and `docs` sections from `[project.optional-dependencies]`. **Keep the `llm` section** — it is a production optional group and must stay in the lock file so `uv sync --extra llm` resolves reproducibly.
   - From the project root (Python 3.13 available to uv):

   ```bash
   uv lock
   ```

   - Restore the `dev` and `docs` sections to `pyproject.toml`.

   When bumping **`sherpa-onnx`**, also update the matching pin in `[tool.uv] dependency-metadata` (`requires-dist` for `sherpa-onnx-core`). The PyPI sdist omits that dependency; wheels declare it, but uv resolves from sdist metadata unless this override is present.

3. **Verify** — check that `uv.lock`:
   - Contains only production dependencies when dev/docs sections were removed for the lock step (no pytest, black, sphinx in that intermediate lock if you inspect before restoring extras).
   - Contains `llama-cpp-python` and `huggingface-hub` with `marker = "extra == 'llm'"`.
   - Python version is constrained to the 3.13 minor series in `requires-python`.

4. **Install locally**:

   ```bash
   uv sync --frozen
   uv sync --extra dev --frozen
   ```

5. **Update third-party notices** (when dependencies change):
   - Run `uv run python scripts/licensing/compare_lock_to_notices.py` to list missing packages, version mismatches, and stale folders.
   - Run `uv run python scripts/licensing/sync_pypi_notices.py` to drop orphan dirs, rename preserved manual trees to the pinned `{name}-{version}` folder, and fetch PyPI license files only for packages that have no notice folder yet.
   - Optional: `uv run python scripts/licensing/sync_pypi_notices.py --refresh PACKAGE [PACKAGE ...]` re-downloads license material from PyPI for named distributions (overwrites that package’s folder). Use when upstream license files changed or a preserved tree should be replaced by the sdist/wheel contents.
   - Run `uv run python scripts/licensing/validate_notice_dirs.py` to confirm every locked dependency (except the first-party package) has license material under `NOTICES/PYPI_LICENSES/`.
   - Do **not** run `scripts/licensing/fetch_licenses.py` wholesale; it deletes `NOTICES/PYPI_LICENSES` and drops manually curated license trees.
   - If sync reports **metadata-only** or **missing** packages, add or restore full license text manually (for example from the project’s upstream repository or tag when the wheel/sdist omits a `LICENSE` file).
   - Update `NOTICES/NOTICE.txt` when bundled assets or runtime-downloaded models (not PyPI packages) change; keep `DISCLAIMER.md` aligned for user-facing summaries.

Reproducible installs for releases and CI use `uv.lock` with `uv sync --frozen` (and the appropriate `--extra` flags). There are no separate `requirements*.txt` files.
