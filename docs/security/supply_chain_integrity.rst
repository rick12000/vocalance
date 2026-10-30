Supply Chain Integrity
########################

Python Libraries
================

All Python dependencies are sourced exclusively from
`PyPI <https://pypi.org>`_ over HTTPS. No private indexes or mirrors are used.

`uv <https://github.com/astral-sh/uv>`_ manages the dependency tree. ``uv.lock``
records the exact version and SHA-256 hash of every package in the full
transitive closure. ``uv sync --frozen`` verifies each downloaded wheel against
its recorded hash before installation — a mismatch is a hard error.

UV Bootstrap
============

The installation script ``setup.ps1`` bootstraps ``uv`` by downloading the official
``uv-{arch}-pc-windows-msvc.zip`` binary archive from GitHub releases. The
archive is never executed. ``uv.exe`` is extracted from it and placed inside
the Vocalance install tree at ``%LOCALAPPDATA%\Programs\Vocalance\tools\uv.exe``.
No system-wide UV installation occurs.

Before extraction, the script verifies the archive's SHA-256 against a value
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

The AI model feature is optional (the installation script will ask the user
whether they want to enable it — see :ref:`security/installation_uninstallation:Installation flow`) and only a subset of dictation modes require it.
When enabled, models are downloaded at first
launch from `Hugging Face <https://huggingface.co>`_.

Three controls are applied in sequence to every download.

Model allowlist
---------------

Only six models are permitted. The allowlist is hard-coded as a frozen Pydantic
model in
`vocalance/app/config/app_config.py <https://github.com/rick12000/vocalance/blob/main/vocalance/app/config/app_config.py>`_:

.. code-block:: python

   class LocalLLMArtifact(BaseModel):
       model_config = ConfigDict(frozen=True)
       id: str
       repo_id: str
       gguf_filenames: tuple[str, ...]
       gguf_sha256: Dict[str, str]  # filename → expected SHA-256

The six permitted models are pinned GGUF builds of these checkpoints:

- ``qwen3.5-2b-q4km`` — Qwen3.5 2B (default), Q4_K_M from ``unsloth/Qwen3.5-2B-GGUF``
- ``qwen3-4b-instruct-2507-q4km`` — Qwen3 4B Instruct 2507, Q4_K_M from ``unsloth/Qwen3-4B-Instruct-2507-GGUF``
- ``granite-4.2-3b-q4km`` — Granite 4.2 3B, Q4_K_M from ``ibm-granite/granite-4.2-3b-GGUF``
- ``qwen3.5-4b-q4km`` — Qwen3.5 4B, Q4_K_M from ``unsloth/Qwen3.5-4B-GGUF``
- ``ministral-3-8b-iq3xs`` — Ministral 3 8B Instruct, IQ3_XS from ``bartowski/mistralai_Ministral-3-8B-Instruct-2512-GGUF``
- ``qwen3-8b-iq3m`` — Qwen3 8B, IQ3_M from ``bartowski/Qwen_Qwen3-8B-GGUF``

Any model ID not in this list is rejected by the ``LLMConfig`` field validator
before a download is ever attempted.

Optional **local endpoint** mode (user opt-in in Settings) sends chat completions to
an OpenAI-compatible HTTP API on **loopback only** (``localhost``, ``127.0.0.1``, or
``::1``). No API key field is offered. The server (e.g. Ollama or ``llama-server``)
is outside Vocalance's SHA-256 allowlist; the user is responsible for what runs on
that port. Shipped GGUF downloads are skipped at startup when endpoint mode is active.

Redirect pinning
----------------

Hugging Face CDN downloads are often served via HTTP redirect. Before following
any redirect, the downloader validates that the target hostname is present in a
hard-coded allowlist of trusted Hugging Face infrastructure domains:

.. code-block:: python

   _TRUSTED_HF_HOSTS = frozenset({
       "huggingface.co",
       "cdn-lfs.huggingface.co",
       "cdn-lfs-us-1.huggingface.co",
       "hf.co",
       "cdn-lfs.hf.co",
       "cdn-lfs-us-1.hf.co",
       "cdn-lfs-eu-1.hf.co",
       "xethub.hf.co",
       "cas-bridge.xethub.hf.co",
   })

   def _validate_hf_redirect(response: httpx.Response) -> None:
       if response.is_redirect:
           location = response.headers.get("location", "")
           host = urlparse(location).hostname or ""
           trusted = host in _TRUSTED_HF_HOSTS or any(
               host.endswith(f".{h}") for h in _TRUSTED_HF_HOSTS
           )
           if not trusted:
               raise ValueError(f"Blocked redirect to untrusted host: {host!r}")

The allowlist covers both the legacy ``huggingface.co`` CDN subdomains and the
current ``hf.co`` CDN (``cdn-lfs-us-1.hf.co``, ``cdn-lfs-eu-1.hf.co``,
``cdn-lfs.hf.co``) as well as the XetHub storage bridge
(``cas-bridge.xethub.hf.co``) introduced in February 2025. A redirect to any
host not in this set, or not a subdomain of a listed host, aborts the download.

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

TLS certificate stores
======================

Following the eMudhra / emSign CA incident, the project pins a minimum
``certifi`` version in ``pyproject.toml``. Git hooks are run with
`prek <https://github.com/j178/prek>`_, which provisions Python hook
environments through ``uv`` using the project lockfile rather than ad hoc
per-hook ``pip`` bootstraps.

Bundled Models
==============

The YAMNet sound classifier and Vosk speech recognition model are embedded
directly in the release zip and are never downloaded separately (no need for external integrity validation).
