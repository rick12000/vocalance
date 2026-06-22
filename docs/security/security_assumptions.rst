Security Assumptions
#####################

Below we detail what is and isn't included in Vocalance's threat model.

What We Protect Against
========================

Supply-Chain Integrity
----------------------

Every third-party asset downloaded at install or runtime is hash-verified
before use. Full details of each control are documented in
:doc:`supply_chain_integrity`.

Hash verification is not applied to the Vocalance release zip itself because
both the zip and any hash embedded in ``setup.ps1`` originate from the same
GitHub release. If that release were compromised, both would be replaced
simultaneously, making the check self-referential and worthless.

Examples:

.. list-table::
   :widths: 38 62
   :header-rows: 1
   :class: uniform-rows

   * - Asset
     - Control
   * - Python packages
     - SHA-256 per wheel in ``uv.lock``; ``uv sync --frozen`` refuses to install
       a wheel whose hash does not match.
   * - ``uv`` binary
     - SHA-256 hard-coded in ``setup.ps1``, computed offline against Astral's
       independent GitHub release; mismatch aborts setup and deletes the archive.
   * - AI model files (``.gguf``)
     - SHA-256 per file hard-coded in the allowlist in ``app_config.py``;
       mismatch deletes the file and raises ``IntegrityError``.

Tampered Local Storage
-----------------------

User data files are plain JSON in ``%APPDATA%\Vocalance\``,
writable by any process with standard user access. The application defends at
the ingestion layer:

- **Pydantic deserialisation** rejects structurally invalid data and
  out-of-bounds numeric values.
- **Hotkey allowlist** drops any custom command that fails validation;
  non-hotkey action types are also dropped (see :doc:`input_validation`).
- **Alias sanitiser** rejects any key or value containing control characters,
  preventing terminal injection via pasted text.
- **Settings allowlist** discards any configuration key not in
  ``ALLOWED_USER_SETTING_PATHS`` — arbitrary keys cannot reach internal
  parameters.
- **Fail-safe defaults** — ``CommandsData`` and ``SoundMappingsData``
  validation failures return an empty safe default and emit a user-visible
  corruption warning.

A malicious actor with write access to the data directory cannot use that access
to make the application execute arbitrary key sequences or bypass validation.

What is Out of Scope
=====================

Audio-Input Spoofing
--------------------

Vocalance's command and dictation pipelines assume the audio input is trustworthy.
A sufficiently privileged attacker on the same machine could create a virtual audio device,
set it as the system default input, and feed arbitrary audio to trigger commands
or dictation.

This vector is explicitly out of scope. Mounting it requires administrator-level
access or a kernel/driver-level malicious component. An attacker at that
privilege level already possesses:

- The ability to run arbitrary scripts and executables.
- Full read/write access to all files on the machine.

Therefore, such an attacker would have nothing to gain by hijacking Vocalance.
