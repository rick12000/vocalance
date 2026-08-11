# Dependency Updates

## Dependency Groups

`pyproject.toml` defines four dependency groups:

| Group | Purpose | Install command |
|-------|---------|-----------------|
| *(base)* | Core runtime dependencies | `uv sync` |
| `llm` | Optional AI features: smart dictation and AI text editing. Requires Microsoft C++ Build Tools. | `uv sync --extra llm` |
| `dev` | Development tooling: pytest, black, flake8, sphinx, pre-commit | `uv sync --extra dev` |
| `docs` | Documentation build tools | `uv sync --extra docs` |

`llama-cpp-python` and `huggingface-hub` belong exclusively to the `llm` group and must not appear in the base `dependencies` list. In `requirements.txt` they appear as uncommented entries below a comment indicating they are optional LLM extras — this is intentional; `requirements.txt` documents all dependencies including optional ones, and the comment makes their status clear.

## Process

1. **Update `pyproject.toml`**:
   - Base runtime packages go under `[project] dependencies`
   - LLM-only packages go under `[project.optional-dependencies] llm`
   - Dev/tooling packages go under `[project.optional-dependencies] dev`

2. **Update `requirements.txt`** — must exactly mirror the base `[project] dependencies` list. LLM packages are listed at the bottom as commented-out references only.

3. **Update dev dependencies** — if modifying `[project.optional-dependencies] dev`, also update `requirements-dev.txt` to match. Keep `requirements-dev.txt` starting with `-r requirements.txt` followed by additional dev packages.

4. **Create temporary environment**:
   ```bash
   conda create -n temp_lock_env python=3.13.9 -y
   conda activate temp_lock_env
   pip install uv
   ```

5. **Generate clean `uv.lock`**:
   - Temporarily remove only the `dev` and `docs` sections from `[project.optional-dependencies]` in `pyproject.toml`. **Keep the `llm` section** — it is a production optional group and must be included in the lock file so that `uv sync --extra llm` resolves reproducibly.
   - Run `uv lock` in the temporary environment from the project root.
   - Restore the `dev` and `docs` sections to `pyproject.toml`.

6. **Generate `requirements-dist.txt`**:
   ```bash
   uv export --no-dev --format requirements-txt -o requirements-dist.txt
   ```
   This creates a fully resolved, pinned requirements file suitable for reproducible deployments. The `--no-dev` flag ensures dev/docs tools are excluded. LLM packages will be absent unless you add `--extra llm`.

7. **Clean up** — remove temporary environment:
   ```bash
   conda remove -n temp_lock_env --all -y
   ```

8. **Verify** — check that `uv.lock`:
   - Contains only production dependencies (no dev tools like pytest, black, sphinx).
   - Contains `llama-cpp-python` and `huggingface-hub` with `marker = "extra == 'llm'"` so they are only installed when the `llm` extra is requested.
   - Python version is pinned to `3.13.9` in `requires-python`.
