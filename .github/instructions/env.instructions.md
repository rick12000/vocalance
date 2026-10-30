---
applyTo: "**"
---

# Python environment

Use this virtual environment for Python commands. If this repository is only a Python application, use it for every command.

## Environment

Directory: `<repo-root>\.venv`

## Activate

From the repository root:

PowerShell:

    .\.venv\Scripts\Activate.ps1

cmd:

    call .venv\Scripts\activate.bat

## Run

Run commands with either activation (above), or without activating:

    uv run python main.py
    uv run pytest tests/

`uv run` automatically uses `.venv` and synchronizes it with `pyproject.toml` and `uv.lock`.

## Install

From the repository root, when dependencies change:

    uv pip install -e . --python .venv\Scripts\python.exe

Use `uv pip install -r requirements.txt --python .venv\Scripts\python.exe` when the only manifest is `requirements.txt`. Skip reinstall if the editable install is already done. If install fails on version conflicts, update `pyproject.toml`, `setup.py`, `setup.cfg`, or `requirements.txt`, retry, and report those edits in the final reply.
