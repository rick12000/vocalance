Security Guidance
#################

A series of developer-oriented documents describing the security practices
adopted in the Vocalance codebase. Intended for developers, contributors, and
technically informed users.

.. admonition:: No guarantees

   These documents are **purely informational** and do not constitute a legally
   binding security policy. Vocalance is provided *as-is* under the
   `GNU General Public License v3 <https://github.com/rick12000/vocalance/blob/main/LICENSE.txt>`_,
   which explicitly disclaims all warranties.

Topics
======

- :doc:`releases` — CI/CD pipeline steps from merge to immutable GitHub release,
  including linting, security scanning, unit tests, privacy guards, release
  packaging, and SHA-256 checksum generation.

- :doc:`privacy` — Network posture after installation, the two opt-in logging
  mechanisms and what each records, and the user data persisted across sessions.

- :doc:`third_party_dependencies` — Python library sourcing and hash pinning via
  ``uv``, the UV bootstrap integrity check, and the three-layer integrity control
  applied to AI models downloaded from Hugging Face.

- :doc:`installation_uninstallation` — What ``setup.ps1`` does step-by-step,
  the privilege model, where files land, and how ``cleanup.ps1`` removes all
  application data.

- :doc:`input_validation` — Settings bounds enforcement, storage-layer ingestion
  validators, hotkey allowlist and single-combo enforcement, and alias
  control-character blocking.

- :doc:`security_assumptions` — Explicit threat-model boundaries: cryptographic
  controls in place, tampered-storage defences, and why audio-input spoofing is
  out of scope.
