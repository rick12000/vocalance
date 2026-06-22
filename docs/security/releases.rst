Releases
########

.. sectnum::

All releases are produced by the CI/CD pipeline defined in
`.github/workflows/ci-cd.yml <https://github.com/rick12000/vocalance/blob/main/.github/workflows/ci-cd.yml>`_.

.. mermaid::

   flowchart TD
       Merge[Push to main] --> Lint[1. Lint]
       Lint --> Sec[2. Security scan]
       Sec --> Tests[3. Unit tests]
       Merge --> LogOff[4a. Logging default off]
       Merge --> ActOff[4b. Activity tracking default off]
       Tests --> Release[5. Draft release]
       LogOff --> Release
       ActOff --> Release
       Release --> Publish[Developer publishes → immutable]

1. Lint
=======

Runs ``pre-commit run --all-files``, enforcing Black and flake8 across the full
working tree. Any violation blocks the rest of the pipeline.

2. Security Scan
================

`Trivy <https://github.com/aquasecurity/trivy>`_ scans the repository filesystem
for known package vulnerabilities, committed secrets, and misconfigurations. The
action is pinned to a commit SHA rather than a mutable tag:

.. code-block:: text

   aquasecurity/trivy-action@57a97c7e7821a5776cebc9bb87c984fa69cba8f1

Results are uploaded to the GitHub Security tab as SARIF. Any ``CRITICAL`` or
``HIGH`` finding fails the job and blocks the pipeline.

3. Unit Tests
=============

Installs ``.[dev,llm]`` and runs pytest under ``xvfb-run`` (virtual framebuffer
required for Qt). Integration, slow, memory, and stress markers are excluded.

4. Privacy Guards
=================

Two jobs run in parallel and are independently required before a release is
created. Both use Python's ``ast`` module to parse source files at the AST level
— not by importing them — so they cannot be fooled by conditional logic or
runtime overrides.

**4a. Logging default off** — asserts ``LoggingConfigModel.enable_logs`` defaults
to ``False`` in ``vocalance/app/config/logging_config.py``.

**4b. Activity tracking default off** — asserts ``ActivityTrackingConfig.enabled``
defaults to ``False`` in ``vocalance/app/config/app_config.py``.

A build that flips either default to ``True`` fails here and cannot produce a
release.

5. Draft Release
================

Runs on push to ``main`` only. If the current version is newer than the latest
published release and no release for it already exists, the job:

- Assembles ``vocalance-v{VERSION}.zip``:

  .. code-block:: text

     vocalance/   application source
     vocalance.py entry point
     pyproject.toml
     uv.lock      pinned, hashed dependency tree
     README.md
     DISCLAIMER.md
     NOTICES/     third-party licence disclosures

- Computes ``vocalance-v{VERSION}.zip.sha256`` via ``sha256sum``.
- Creates a **draft** GitHub release tagged ``v{VERSION}``, attaching:

  - ``vocalance-v{VERSION}.zip``
  - ``vocalance-v{VERSION}.zip.sha256``
  - ``setup.ps1`` — standalone installer script
  - ``cleanup.ps1`` — standalone uninstaller script

A developer then reviews the draft, writes the release notes, and publishes. Once
published, the release is **immutable**: it cannot be edited or deleted. The zip,
its checksum, and the bootstrapping scripts are the canonical, unalterable artifacts
for that version.
