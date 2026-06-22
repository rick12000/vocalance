Installation and Uninstallation
################################

Installation
============

The application can either be built from source manually (as detailed in the README) or
by using the dedicated
`setup.ps1 <https://github.com/rick12000/vocalance/releases/latest/download/setup.ps1>`_
installation script (which is bundled with each release). Below we detail the workflow of the script.

Privilege model
---------------

In accordance with the principle of least privilege, the script runs as the current user. No administrator privileges are
requested or required. The application is installed at ``%LOCALAPPDATA%\Programs\Vocalance\``.

Installation flow
-----------------

1. **Bootstrap uv** — downloads the ``uv-{arch}-pc-windows-msvc.zip`` binary
   archive from GitHub releases for the pinned ``$UV_VERSION``, verifies its
   SHA-256 against the hard-coded ``$UV_ZIP_SHA256`` value before extracting.
   A mismatch deletes the archive and aborts. ``uv.exe`` is placed in
   ``%LOCALAPPDATA%\Programs\Vocalance\tools\uv.exe``; no system-wide UV
   installation occurs and ``PATH`` is not modified. See
   :ref:`security/supply_chain_integrity:UV Bootstrap` for the full verification
   procedure.

2. **Download Vocalance release** — fetches ``vocalance-v{VERSION}.zip``
   from the immutable GitHub releases page for the pinned version and unpacks it to
   ``%LOCALAPPDATA%\Programs\Vocalance\app\``.

3. **Create virtual environment** — ``uv venv --python 3.13.9``.

4. **Install dependencies** — prompts the user on whether to include LLM features,
   then runs ``uv sync --frozen`` if features are excluded and ``uv sync --frozen --extra llm`` otherwise.
   ``--frozen`` enforces ``uv.lock`` with per-package hash verification (see
   :ref:`security/supply_chain_integrity:Python Libraries`).

5. **Create Start Menu shortcut** — creates a shortcut to the application's entry point ``pythonw.exe vocalance.py``.

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
   * - ``%APPDATA%\Vocalance\``
     - All runtime-written data: configuration, marks, aliases, commands,
       activity logs, LLM model files, developer log files. Created by the
       application on first launch.


Uninstallation
==============

An uninstallation script
`cleanup.ps1 <https://github.com/rick12000/vocalance/releases/latest/download/cleanup.ps1>`_
is also provided with each release.
It runs as the current user (no elevation) and removes the contents of:

1. ``%LOCALAPPDATA%\Programs\Vocalance\`` — source, virtual environment, bundled
   uv binary.
2. ``%APPDATA%\Vocalance\`` — all user-configured state.
3. The Start Menu shortcut.

After cleanup, no Vocalance files remain on the machine. Third-party package
caches (uv, PyPI) and system prerequisites installed independently by the user
(e.g. Microsoft C++ Build Tools) are not touched.
