# Dependency Updates

## Process

1. **Update `pyproject.toml`** - Add or modify packages under `[project] dependencies`

2. **Update `requirements.txt`** - Ensure it exactly matches the dependencies from `pyproject.toml`

3. **Update dev dependencies** - If modifying `[project.optional-dependencies] dev`, also update `requirements-dev.txt` to match. Keep `requirements-dev.txt` starting with `-r requirements.txt` followed by additional dev packages.

4. **Create temporary environment**:
   ```bash
   conda create -n temp_lock_env python=3.13.9 -y
   conda activate temp_lock_env
   pip install uv
   ```

5. **Generate clean `uv.lock`**:
   - Remove `[project.optional-dependencies]` section from `pyproject.toml`
   - Run `uv lock` in the temporary environment
   - Restore `[project.optional-dependencies]` section to `pyproject.toml`

6. **Generate `requirements-dist.txt`**:
   ```bash
   uv pip compile pyproject.toml -o requirements-dist.txt
   ```
   This creates a fully resolved, pinned requirements file suitable for reproducible deployments.

7. **Clean up** - Remove temporary environment:
   ```bash
   conda remove -n temp_lock_env --all -y
   ```

8. **Verify** - Check that `uv.lock` contains only production dependencies (no dev tools like pytest, black, sphinx). Python version is pinned to 3.13.9 in `requires-python`.
