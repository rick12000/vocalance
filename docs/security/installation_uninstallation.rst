Installation and Uninstallation
################################

.. sectnum::

Installation
============

The installer is
`setup.ps1 <https://github.com/rick12000/vocalance/releases/latest/download/setup.ps1>`_,
a PowerShell script for Windows 10/11 distributed as a standalone release asset.

Privilege model
---------------

The script runs entirely as the current user. No administrator privileges are
requested or required. Installation targets ``%LOCALAPPDATA%\Programs\Vocalance\``,
following the same user-space convention used by VS Code, Chrome, and Slack.

This approach enforces the principle of least privilege: the application runs as
the current user and is installed by the current user. No UAC prompt, no
self-elevation, and no ``-ExecutionPolicy Bypass`` flag are ever used.

Installation flow
-----------------

1. **Bootstrap uv** — downloads the ``uv-{arch}-pc-windows-msvc.zip`` binary
   archive from GitHub releases for the pinned ``$UV_VERSION``, verifies its
   SHA-256 against the hard-coded ``$UV_ZIP_SHA256`` value before extracting.
   A mismatch deletes the archive and aborts. ``uv.exe`` is placed in
   ``%LOCALAPPDATA%\Programs\Vocalance\tools\uv.exe``; no system-wide UV
   installation occurs and ``PATH`` is not modified.

2. **Download and extract release zip** — fetches ``vocalance-v{VERSION}.zip``
   from the GitHub releases page for the pinned version and unpacks it to
   ``%LOCALAPPDATA%\Programs\Vocalance\app\``.

3. **Create virtual environment** — ``uv venv --python 3.13.9``.

4. **Install dependencies** — prompts the user whether to include LLM support,
   then runs ``uv sync --frozen`` or ``uv sync --frozen --extra llm``.
   ``--frozen`` enforces ``uv.lock`` with per-package hash verification.

5. **Create Start Menu shortcut** — ``pythonw.exe vocalance.py`` under the
   current user's ``Programs`` folder (no admin required).

File layout
-----------

.. list-table::
   :widths: 45 55
   :header-rows: 1
   :class: uniform-rows

   * - Path
     - Contents
   * - ``%LOCALAPPDATA%\Programs\Vocalance\app\``
     - Source code, ``uv.lock``, bundled models, scripts.
   * - ``%LOCALAPPDATA%\Programs\Vocalance\env\``
     - Python virtual environment.
   * - ``%LOCALAPPDATA%\Programs\Vocalance\tools\uv.exe``
     - Bundled uv binary; scoped to this installation.
   * - ``%APPDATA%\vocalance_voice_assistant_data\``
     - All runtime-written data: configuration, marks, aliases, commands,
       activity logs. Created by the application on first launch.
   * - ``%LOCALAPPDATA%\vocalance_voice_assistant\cache\``
     - Downloaded AI model files; developer log files if logging is enabled.

All paths are within the current user's profile; no system directories are
touched.

Uninstallation
==============

The uninstaller is
`cleanup.ps1 <https://github.com/rick12000/vocalance/releases/latest/download/cleanup.ps1>`_,
distributed as a standalone release asset. It runs as the current user (no elevation) and removes, in order:

1. ``%LOCALAPPDATA%\Programs\Vocalance\`` — source, virtual environment, bundled
   uv binary.
2. ``%APPDATA%\vocalance_voice_assistant_data\`` — all user-configured state.
3. The Start Menu shortcut.

After cleanup, no Vocalance files remain on the machine. Third-party package
caches (uv, PyPI) and system prerequisites installed independently by the user
(e.g. Microsoft C++ Build Tools) are not touched.
