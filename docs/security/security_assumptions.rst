Security Assumptions
#####################

.. sectnum::

This page makes Vocalance's threat model explicit: what the application actively
defends against, and what is considered out of scope.

What We Protect Against
========================

Supply-Chain Integrity
----------------------

Every third-party asset downloaded at install or runtime is hash-verified
before use. The expected hash is committed to this repository and computed
offline — it does not come from the same source as the asset it protects,
so serving a malicious file cannot produce a matching hash.

Hash verification is not applied to the Vocalance release zip itself because
both the zip and any hash embedded in ``setup.ps1`` originate from the same
GitHub release. If that release were compromised, both would be replaced
simultaneously, making the check self-referential and worthless.

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

User data files are plain JSON in ``%APPDATA%\vocalance_voice_assistant_data\``,
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

Vocalance's entire command pipeline — segmentation, classification, recognition,
parsing, and execution — assumes the audio input is trustworthy. A sufficiently
privileged attacker on the same machine could create a virtual audio device,
set it as the system default input, and feed arbitrary audio to trigger commands
or dictation.

This vector is explicitly out of scope. Mounting it requires administrator-level
access or a kernel/driver-level malicious component. An attacker at that
privilege level already possesses:

- The ability to run arbitrary scripts and executables.
- Direct API access to inject keystrokes and hotkeys.
- Full read/write access to all files on the machine.
- The ability to install, modify, or terminate any process.

The marginal capability gained by hijacking Vocalance's command execution is
negligible relative to what that attacker already has. Defending against an
adversary who owns the machine at the driver level is not a tractable goal for
a user-space application — the appropriate controls are OS-level (endpoint
protection, privilege management, driver signing).

.. admonition:: Deployment note

   Vocalance is designed for machines the user trusts and controls. Enterprise
   deployments should apply standard endpoint security controls independently
   of the application.

Post-Download AI Model Substitution
-------------------------------------

``.gguf`` files are hash-verified immediately after download but not on
subsequent loads. An attacker with write access to
``%LOCALAPPDATA%\vocalance_voice_assistant\cache\llm_models\`` could substitute
a model file between the initial verification and a later load. This is accepted
for the same reason as audio spoofing: that level of local access already
provides a broad attack surface independent of Vocalance. The impact is
additionally bounded by what ``llama-cpp-python`` will do with a malformed
binary — a library-level concern outside Vocalance's control.

Known-Vulnerable Pinned Dependencies
--------------------------------------

Pinned package versions and hashes guarantee reproducibility, not the absence
of vulnerabilities. Trivy scanning (see :doc:`releases`) checks for known CVEs
at build time; vulnerabilities disclosed after a release are not automatically
remediated. Users in security-sensitive environments should monitor upstream
advisories for packages listed in ``uv.lock`` and ``pyproject.toml``.
