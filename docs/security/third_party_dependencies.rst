Third-Party Dependencies
########################

.. sectnum::

Python Libraries
================

All Python dependencies are sourced exclusively from
`PyPI <https://pypi.org>`_ over HTTPS. No private indexes or mirrors are used.

`uv <https://github.com/astral-sh/uv>`_ manages the dependency tree. ``uv.lock``
records the exact version and SHA-256 hash of every package in the full
transitive closure. ``uv sync --frozen`` verifies each downloaded wheel against
its recorded hash before installation — a mismatch is a hard error. This applies
both when installing from source and when distributing: ``requirements-dist.txt``
is generated with ``uv export --format requirements-txt`` and carries the same
per-wheel hashes for use with ``pip --require-hashes``.

UV Bootstrap
============

``setup.ps1`` bootstraps ``uv`` by downloading the official
``uv-{arch}-pc-windows-msvc.zip`` binary archive from GitHub releases. The
archive is never executed — ``uv.exe`` is extracted from it and placed inside
the Vocalance install tree at ``%LOCALAPPDATA%\Programs\Vocalance\tools\uv.exe``.
No system-wide UV installation occurs.

Before extracting, the script verifies the archive's SHA-256 against a value
computed offline and hard-coded at development time:

.. code-block:: powershell

   $UV_VERSION    = '0.11.22'
   $UV_ZIP_SHA256 = @{
       'x86_64'  = '<hex digest>'
       'aarch64' = '<hex digest>'
   }

   $actual = (Get-FileHash -Path $uvZipPath -Algorithm SHA256).Hash.ToLower()
   if ($actual -ne $UV_ZIP_SHA256[$arch].ToLower()) {
       Remove-Item -LiteralPath $uvZipPath -Force -ErrorAction SilentlyContinue
       throw "Integrity check failed ..."
   }

The hashes are produced by
`scripts/security/compute_uv_binary_hash.ps1 <https://github.com/rick12000/vocalance/blob/main/scripts/security/compute_uv_binary_hash.ps1>`_
whenever a developer bumps ``$UV_VERSION``. A mismatch deletes the downloaded
archive and aborts setup; ``uv.exe`` is never extracted from an unverified
archive.

AI Models
=========

The AI model feature is optional — only the Smart and Amend dictation modes
require it. Users are prompted at installation time and can add the ``[llm]``
extra independently at any time. When enabled, models are downloaded at first
launch from `Hugging Face <https://huggingface.co>`_.

Three controls are applied in sequence to every download.

Model allowlist
---------------

Only three models are permitted. The allowlist is hard-coded as a frozen Pydantic
model in
`vocalance/app/config/app_config.py <https://github.com/rick12000/vocalance/blob/main/vocalance/app/config/app_config.py>`_:

.. code-block:: python

   class LocalLLMArtifact(BaseModel):
       model_config = ConfigDict(frozen=True)
       id: str
       repo_id: str
       gguf_filenames: tuple[str, ...]
       gguf_sha256: Dict[str, str]  # filename → expected SHA-256

The three permitted models:

- ``qwen2.5-1.5b-q5km`` — Qwen2.5 1.5B Instruct (Q5_K_M)
- ``qwen3-4b-q5km`` — Qwen3 4B (Q5_K_M)
- ``qwen3-8b-q5km`` — Qwen3 8B (Q5_K_M)

Any model ID not in this list is rejected by the ``LLMConfig`` field validator
before a download is ever attempted.

Redirect pinning
----------------

Hugging Face CDN downloads are often served via HTTP redirect. Before following
any redirect, the downloader validates that the target hostname is
``huggingface.co`` or a subdomain:

.. code-block:: python

   def _validate_hf_redirect(response: httpx.Response) -> None:
       if response.is_redirect:
           location = response.headers.get("location", "")
           host = urlparse(location).hostname or ""
           if not (host == "huggingface.co" or host.endswith(".huggingface.co")):
               raise ValueError(f"Blocked redirect to untrusted host: {host!r}")

A redirect to any other domain aborts the download.

SHA-256 verification
--------------------

Each ``gguf_sha256`` entry holds the expected digest of the corresponding
``.gguf`` file, computed offline by a developer using
`scripts/security/compute_llm_hashes.py <https://github.com/rick12000/vocalance/blob/main/scripts/security/compute_llm_hashes.py>`_.
After every completed download:

.. code-block:: python

   actual = self._sha256_of_file(partial_path)
   if actual != expected_sha256.lower():
       os.remove(partial_path)
       raise IntegrityError(
           f"SHA-256 mismatch for {filename!r}: "
           f"expected {expected_sha256} got {actual}"
       )

A mismatch deletes the file and raises ``IntegrityError``, surfaced to the user
as an explicit error. The application will not load a model that failed
verification.

Bundled Models
==============

The YAMNet sound classifier and Vosk speech recognition model are embedded
directly in the release zip and are never downloaded separately. Their integrity
is covered by the release zip's published SHA-256 checksum (see :doc:`releases`),
which users may verify independently.
