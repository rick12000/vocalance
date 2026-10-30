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

3. **Verify** — check that `uv.lock`:
   - Contains only production dependencies when dev/docs sections were removed for the lock step (no pytest, black, sphinx in that intermediate lock if you inspect before restoring extras).
   - Contains `llama-cpp-python` and `huggingface-hub` with `marker = "extra == 'llm'"`.
   - Python version is constrained to the 3.13 minor series in `requires-python`.

4. **Install locally**:

   ```bash
   uv sync --frozen
   uv sync --extra dev --frozen
   ```

Reproducible installs for releases and CI use `uv.lock` with `uv sync --frozen` (and the appropriate `--extra` flags). There are no separate `requirements*.txt` files.
