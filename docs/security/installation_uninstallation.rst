Installation and Uninstallation
################################

.. sectnum::

Installation
============

The installer is
`scripts/bootstrapping/setup.ps1 <https://github.com/rick12000/vocalance/blob/main/scripts/bootstrapping/setup.ps1>`_,
a PowerShell script for Windows 10/11.

Privilege model
---------------

The script self-elevates to administrator via UAC. Admin access is required to
write to ``C:\Program Files\`` and to apply ACLs that prevent standard users
from modifying application files at runtime. The application itself runs as the
current non-admin user.

Installation flow
-----------------

1. **Self-elevate** — re-launches with ``-Verb RunAs`` if not already elevated.

2. **Bootstrap uv** — downloads the ``uv`` installer from
   ``https://astral.sh/uv/install.ps1`` for the pinned ``$UV_VERSION``,
   verifies its SHA-256 against the hard-coded ``$UV_INSTALLER_SHA256`` before
   executing. A mismatch deletes the file and aborts. See
   :doc:`third_party_dependencies`.

3. **Download release zip** — fetches ``vocalance-v{VERSION}.zip`` from the
   GitHub releases page for the pinned version.

4. **Extract and harden** — unpacks to ``C:\Program Files\Vocalance\app``, then
   sets ACLs: Administrators and SYSTEM receive full control; standard Users
   receive read and execute only. The running application process cannot modify
   its own code on disk.

5. **Create virtual environment** — ``uv venv --python 3.13.9``.

6. **Install dependencies** — prompts the user whether to include LLM support,
   then runs ``uv sync --frozen`` or ``uv sync --frozen --extra llm``.
   ``--frozen`` enforces ``uv.lock`` with per-package hash verification.

7. **Create Start Menu shortcut** — ``pythonw.exe vocalance.py`` under
   ``Programs\Vocalance``.

File layout
-----------

.. list-table::
   :widths: 45 55
   :header-rows: 1
   :class: uniform-rows

   * - Path
     - Contents
   * - ``C:\Program Files\Vocalance\app\``
     - Source code, ``uv.lock``, bundled models, scripts.
       Read-only for standard users (ACL-enforced).
   * - ``C:\Program Files\Vocalance\venv\``
     - Python virtual environment.
   * - ``%APPDATA%\vocalance_voice_assistant_data\``
     - All runtime-written data: configuration, marks, aliases, commands,
       activity logs. Created by the application on first launch.
   * - ``%LOCALAPPDATA%\vocalance_voice_assistant\cache\``
     - Downloaded AI model files; developer log files if logging is enabled.

Code and bundled assets are in an admin-only location the application process
cannot modify. All data the application writes goes to per-user directories
where write access does not require elevation.

Uninstallation
==============

The uninstaller is
`scripts/bootstrapping/cleanup.ps1 <https://github.com/rick12000/vocalance/blob/main/scripts/bootstrapping/cleanup.ps1>`_.
It self-elevates via UAC and removes, in order:

1. ``C:\Program Files\Vocalance\`` — source, virtual environment, bundled models.
2. ``%APPDATA%\vocalance_voice_assistant_data\`` — all user-configured state.
3. The Start Menu shortcut.

After cleanup, no Vocalance files remain on the machine. Third-party package
caches (uv, PyPI) and system prerequisites installed independently by the user
(e.g. Microsoft C++ Build Tools) are not touched.
