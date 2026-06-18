# Vocalance — Security Code Review

**Date:** 2026-06-17
**Scope:** Full `vocalance/` application source, build/CI/packaging surface, and bootstrap scripts.
**Method:** Component-by-component static analysis (partitioned by parent folder), mapped to OWASP Top 10 (2021), OWASP LLM Top 10 (2025), CWE, and SEI CERT secure-coding standards. Highest-severity findings were re-verified against source.

---

## 1. Executive Summary

Vocalance is a **local, offline-first desktop voice-control application** (PySide6/Qt) that performs speech-to-text (Vosk/Moonshine), custom sound recognition (YAMNet + a trained classifier), optional LLM-based dictation post-processing (`llama-cpp-python`), and **drives the OS mouse/keyboard via `pyautogui`**. It has no inbound network server. The realistic threat model is therefore:

- **T1 — Local tampering / co-resident malware:** another process or user with write access to the per-user data directory (`%APPDATA%\...`) or app install tree.
- **T2 — Adversarial / accidental audio:** crafted or misrecognized speech and trained sounds that the app turns into OS actions.
- **T3 — Supply chain:** model downloads, dependency resolution, bootstrap scripts, and CI.
- **T4 — Privacy:** dictated content (which may include secrets) leaking to logs or other apps.

Because recognized input is converted directly into **keystrokes, hotkeys, clipboard pastes, and mouse clicks**, the blast radius of an input- or storage-tampering bug is the entire user session — not just the app.

### Severity tally

| Severity | Count |
|----------|-------|
| Critical | 1 |
| High | 6 |
| Medium | 15 |
| Low | 12 |
| Info | 9 |

### Top priorities

1. **[C-01] Untrusted `pickle.load` of sound-model files → arbitrary code execution.**
2. **[H-01] Unvalidated `action_value` strings passed straight to `pyautogui.hotkey/press` → arbitrary keystroke injection.**
3. **[H-02] On-disk command/config JSON is trusted after structural validation only → tampered files drive automation.**
4. **[H-03] Downloaded LLM models have no integrity (hash/signature/size) verification.**
5. **[H-04] Unsanitized dictation text pasted into the focused window.**
6. **[H-05] Bootstrap installs UV via `irm … | iex` (remote code execution, unverified).**
7. **[H-06] `tarfile`/`zipfile` `extractall` without path validation (tar/zip slip).**

### Notable existing controls (good practice already in place)

- No `eval`/`exec`/`os.system`/`shell=True` in application runtime code; command dispatch is a static `if/elif`, not dynamic attribute lookup.
- JSON config uses Pydantic validation; YAML uses `yaml.safe_load` (no unsafe deserialization there).
- Model IDs/repos are a hardcoded allowlist; `selected_model_id` is validated.
- Logging is **disabled by default** and enforced by a CI check.
- `requirements-dist.txt` is hash-pinned; `cleanup.ps1` uses `-LiteralPath`.
- Dictation/LLM display uses **plain-text** Qt widgets (`QPlainTextEdit`), avoiding rich-text injection in the main transcript view.

---

## 2. Component Map

| # | Component (parent folder) | Primary security concern |
|---|---------------------------|--------------------------|
| A | `services/dictation_flow/llm/` | Model download integrity, prompt injection |
| B | `services/command_flow/execution/`, `keyboard_input_service.py`, `dictation_flow/text_input_service.py` | OS keystroke/click injection |
| C | `services/storage/`, `config/` | Deserialization, integrity, file permissions, concurrency |
| D | `services/command_flow/{parsing,management,segmenting,sound_recognition,speech_recognition}/`, `capture/`, `utils/` | `pickle` RCE, input validation, resource exhaustion |
| E | `scripts/`, `.github/workflows/`, `pyproject.toml`, requirements | Supply chain, remote execution, archive extraction |
| F | `ui/`, `lifecycle/`, `events/` | Markup injection, concurrency, log privacy |

---

## 3. Findings

### CRITICAL

---

#### C-01 — Arbitrary code execution via `pickle.load` of user-writable sound-model files

- **Severity:** Critical (local code execution; T1)
- **CWE:** CWE-502 (Deserialization of Untrusted Data)
- **Mapping:** OWASP A08:2021 Software & Data Integrity Failures; SEI CERT SER12-J / POS35-C
- **Component:** D — `services/command_flow/sound_recognition/sound_recognizer.py`

The sound recognizer deserializes its label list and feature scaler with `pickle.load`, reading from files located in the user-writable app-data directory (`storage_config.sound_model_dir`). These files are loaded automatically at recognizer initialization.

```224:229:vocalance/app/services/command_flow/sound_recognition/sound_recognizer.py
            with self._model_lock:
                self.embeddings = np.load(embeddings_path)
                with open(labels_path, "rb") as f:
                    self.labels = pickle.load(f)
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
```

The matching writer also uses `pickle.dump` (`_save_model_files_sync`, lines 265–272).

- **Impact:** Any process or user able to replace `labels.joblib` / `scaler.joblib` (T1 — same-user malware, a malicious backup/sync, or a shared profile) achieves **arbitrary Python code execution inside Vocalance** at startup. Because Vocalance can synthesize keystrokes and mouse actions, this is a full-session compromise primitive. `pickle` executes `__reduce__` during load — no user interaction beyond launching the app is required.
- **Recommendation:**
  - Stop using `pickle` for persisted artifacts. Store `labels` as JSON and the scaler’s parameters (mean/scale arrays) as `.npy`/JSON.
  - If a binary format is unavoidable, verify an HMAC/signature over the model bundle (key stored outside the bundle) before loading, and fail closed on mismatch.
  - Treat the entire `sound_model_dir` as untrusted input; rebuild from training data rather than trusting on-disk blobs.
- **Note:** `embeddings.npy` is loaded via `np.load` without `allow_pickle=True`, which is safe for standard arrays; the YAMNet `tf.saved_model.load` (line 304) and other ML model loads are a lower-severity integrity concern (see I-03).

---

### HIGH

---

#### H-01 — Unvalidated key/hotkey strings injected directly into PyAutoGUI

- **Severity:** High (T1/T2)
- **CWE:** CWE-20 (Improper Input Validation), CWE-77-adjacent (command/keystroke composition)
- **Mapping:** OWASP A04:2021 Insecure Design; SEI CERT IDS00-J
- **Component:** B — `services/command_flow/execution/automation_service.py`

`action_value` is split and passed verbatim to `pyautogui.hotkey`, `pyautogui.press`, and a key-sequence executor, with no allowlist or blocklist of permitted keys/combinations:

```43:71:vocalance/app/services/command_flow/execution/automation_service.py
    def create_action_function(self, action_type: ActionType, action_value: str) -> Optional[Callable[[], None]]:
        if action_type == "hotkey":
            keys = [k.strip() for k in action_value.replace(" ", "+").split("+")]
            return lambda: pyautogui.hotkey(*keys)
        if action_type == "key":
            return lambda: pyautogui.press(action_value)
        if action_type == "key_sequence":
            key_list = [k.strip() for k in action_value.split(",")]
            return lambda: self.execute_key_sequence(key_list)
        ...
    def execute_key_sequence(self, key_list: list[str]) -> None:
        for combo in key_list:
            if "+" in combo:
                pyautogui.hotkey(*[k.strip() for k in combo.split("+")])
            else:
                pyautogui.press(combo.strip())
```

`click` and `scroll` are correctly constrained to fixed maps/literals, but `hotkey`, `key`, and `key_sequence` accept any value.

- **Impact:** Combined with H-02 (untrusted storage) or via the custom-command UI, a short voice phrase can be mapped to dangerous combinations (`win+r`, `ctrl+shift+esc`, `alt+f4`) or multi-step sequences that open a Run dialog, launch a shell, or destroy work — all triggered by a single utterance.
- **Recommendation:** Validate every token in `action_value` against a strict allowlist of known-safe key names, and a blocklist of dangerous combinations, at **persist time and load time** (not only execution). For custom commands, restrict the UI to a curated key picker rather than free text.

---

#### H-02 — On-disk command & configuration JSON trusted after structural validation only

- **Severity:** High (T1)
- **CWE:** CWE-353 (Missing Support for Integrity Check), CWE-15 (External Control of Configuration)
- **Mapping:** OWASP A08:2021; SEI CERT FIO01-J
- **Component:** C/D — `services/storage/storage_models.py`, `storage_service.py`, `parsing/command_projection.py`, `parsing/text_command_parse.py`

`CommandsData.custom_commands` is loaded and `model_validate`’d for *shape* only; `action_value` is an unconstrained string. The projection merges custom commands first (so they override built-ins), and the parser feeds them straight into execution:

```54:60:vocalance/app/services/storage/storage_models.py
class CommandsData(StorageData):
    custom_commands: Dict[str, AutomationCommand] = Field(
        default_factory=dict, description="User-defined custom commands mapped by phrase"
    )
```

```188:199:vocalance/app/services/command_flow/parsing/text_command_parse.py
    if normalized_text in action_map:
        spec = action_map[normalized_text]
        return ExactMatchCommand(
            command_key=normalized_text,
            action_type=spec.action_type,
            action_value=spec.action_value,
```

The UI add path (`command_management_service.add_hotkey`) validates only the *phrase* (collisions/protected terms), not the hotkey value, and direct JSON edits bypass even that.

- **Impact:** A T1 attacker editing `custom_commands.json` maps an everyday phrase to an arbitrary `key_sequence`/`hotkey` (see H-01) with no integrity check on load. The same pattern applies to `aliases.json` and `agentic_prompts.json` (text injected into dictation output / LLM system prompt — see A-findings).
- **Recommendation:** Add semantic validation of `AutomationCommand.action_value` at the storage boundary; consider HMAC-signing security-relevant JSON written by the app and refusing/quarantining files that fail verification.

---

#### H-03 — No integrity verification of downloaded LLM model files

- **Severity:** High (T3)
- **CWE:** CWE-494 (Download of Code Without Integrity Check), CWE-345 (Insufficient Verification of Authenticity)
- **Mapping:** OWASP A08:2021; SEI CERT supply-chain integrity
- **Component:** A — `services/dictation_flow/llm/llm_model_downloader.py`, `config/app_config.py`

`LocalLLMArtifact` records only `repo_id` / `gguf_filenames` — no expected SHA-256, size, or pinned revision. Download "success" is *file exists and size > 0*; there is no `downloaded == content-length` check and no revision pin:

```122:144:vocalance/app/services/dictation_flow/llm/llm_model_downloader.py
                    with open(partial_path, "wb") as out:
                        for chunk in response.iter_bytes(chunk_size=_CHUNK_BYTES):
                            ...
                            if chunk:
                                out.write(chunk)
                                downloaded += len(chunk)
            if not os.path.exists(partial_path) or os.path.getsize(partial_path) == 0:
                logger.error("Stream download produced empty file")
                return None
            ...
            shutil.move(partial_path, final_path)
```

```169:175:vocalance/app/services/dictation_flow/llm/llm_model_downloader.py
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=temp_download_dir,
                local_dir_use_symlinks=False,
                resume_download=False,
            )
```

- **Impact:** A swapped/compromised Hugging Face artifact, a same-filename change on `main`, an on-path substitution (despite TLS), or a truncated transfer yields an unverified multi-GB GGUF that is later parsed by native `llama.cpp` (`use_mmap`) — an attractive native-parser attack surface. The same model file is also re-loaded later with only an existence/size check (`llm_service.py` 244–252), so local tampering (T1) is undetected too.
- **Recommendation:** Pin `expected_sha256`, `expected_size`, and `revision` per artifact in the allowlist. Verify the hash on the temp file **before** atomic rename, and re-verify at load time; reject on mismatch and prompt re-download. Enforce a per-artifact max size. Assert `downloaded == total` when `content-length > 0`.
- **Note:** Transport is HTTPS with default verification (`httpx` `verify=True`, `hf_hub_url` https); the gap is integrity, not encryption.

---

#### H-04 — Unsanitized dictation text injected into the focused application

- **Severity:** High (T2/T4)
- **CWE:** CWE-74 (Improper Neutralization of Special Elements in Output)
- **Mapping:** OWASP A03:2021 Injection; SEI CERT STR02
- **Component:** B — `dictation_flow/text_input_service.py`, `dictation_flow/postprocess/segment_text.py`

Speech-derived text is cleaned only by collapsing `"..."`; control characters and newlines are not stripped, then it is copied to the clipboard and pasted with `Ctrl+V` (or typed) into whatever window has focus:

```6:12:vocalance/app/services/dictation_flow/postprocess/segment_text.py
def clean_dictation_text(text: str, add_trailing_space: bool = True) -> str:
    if not text:
        return ""
    cleaned: str = re.sub(r"\.\.\.", " ", text)
    ...
    return cleaned
```

```93:96:vocalance/app/services/dictation_flow/text_input_service.py
        if self.config.use_clipboard:
            success: bool = await self.input_service.run(self.paste_clipboard, cleaned_text)
        else:
            success = await self.input_service.run(self.type_text, cleaned_text)
```

- **Impact:** Misrecognized or adversarial audio becomes OS-level text injection. Pasting newline-bearing content into a terminal, browser address bar, or an app where Enter submits can execute unintended commands or send unintended messages. Dictation also frequently contains secrets (passwords spoken aloud), compounding T4.
- **Recommendation:** Sanitize before injection (strip/normalize `\r\n\t` and other control chars; enforce a max length). Offer a "confirm before paste" mode and/or refuse to inject when the foreground process is a terminal or elevated prompt.

---

#### H-05 — Bootstrap installs UV via remote `irm | iex` without integrity verification

- **Severity:** High (T3)
- **CWE:** CWE-494 (Download of Code Without Integrity Check), CWE-693 (Protection Mechanism Failure)
- **Mapping:** OWASP A08:2021; SLSA build-integrity; supply-chain
- **Component:** E — `scripts/bootstrapping/setup.ps1`

```69:70:scripts/bootstrapping/setup.ps1
    Write-Host 'Installing UV...'
    powershell.exe -ExecutionPolicy Bypass -NoProfile -Command "irm https://astral.sh/uv/install.ps1 | iex"
```

Remote script is fetched and immediately executed with `-ExecutionPolicy Bypass`; there is no version pin, checksum, or signature check. The same script later `git clone`s an overridable `$CloneUrl` (no allowlist, no commit/tag pin) and creates a Start-Menu shortcut that runs the cloned, unsigned `vocalance.py`.

- **Impact:** A compromise of (or MITM against) `astral.sh`, or a maliciously supplied `-CloneUrl`, results in arbitrary code execution on the installing machine. The interactive yes/no prompt limits drive-by abuse but not upstream/MITM compromise.
- **Recommendation:** Install UV from a version-pinned release artifact verified by SHA-256 (or vendor a known-good binary); avoid piping live remote content into `iex`. Pin the clone to a signed tag/commit and validate `$CloneUrl` against an allowlist. For end-user distribution, ship a signed installer.

---

#### H-06 — Archive extraction without path traversal validation (tar/zip slip)

- **Severity:** High (T3, maintainer/CI machine)
- **CWE:** CWE-22 (Improper Limitation of a Pathname to a Restricted Directory)
- **Mapping:** OWASP A01:2021 Broken Access Control
- **Component:** E — `scripts/licensing/fetch_licenses.py`

```147:155:scripts/licensing/fetch_licenses.py
                    with tarfile.open(tar_path, "r:gz") as tar:
                        tar.extractall(extract_dir)
                    tar_path.unlink()
                elif package.sdist_url.endswith(".zip"):
                    ...
                    with zipfile.ZipFile(zip_path, "r") as zf:
                        zf.extractall(extract_dir)
```

`extractall` is called on downloaded sdists/wheels with no member-path sanitization, and the archives are fetched via `urlopen` with **no hash check** against `uv.lock`.

- **Impact:** A malicious or compromised package archive can write outside the temp dir (e.g. `..\..\Startup\evil.ps1`) on the maintainer/CI machine that runs the licensing tool — a path to persistence or build-system compromise.
- **Recommendation:** Use `tar.extractall(..., filter="data")` (Python 3.12+) or validate each member resolves under `extract_dir` before extraction; do the same for zip. Verify downloaded artifacts against the `uv.lock` hashes (or delegate to `uv pip download` / `pip download --require-hashes`).

---

### MEDIUM

> Condensed; each entry: issue · location · CWE/OWASP · fix.

- **M-01 — Silent fallback to empty defaults on corrupt/tampered config.** `storage_service.py` 102–113 swallows `ValidationError`/read errors and returns `model_type()`, masking tampering and silently dropping security-relevant state. CWE-754; OWASP A08. → Distinguish missing-file from corruption; quarantine bad files and surface an integrity alert.
- **M-02 — Prompt injection in dictation/amend flows.** `llm_service.py` 290–310 mixes untrusted STT text and clipboard content with system instructions using spoofable `--- END ---` delimiters; output is injected elsewhere via keyboard/clipboard. CWE-74; OWASP LLM01:2025. → Structurally separate instructions from data, escape delimiters, validate output, "confirm before paste" for amend.
- **M-03 — User-editable agentic prompts placed in the system role unbounded.** `agentic_prompt_service.py` 77–88 + `llm_service.py` 292. CWE-74; OWASP LLM01. → Length/character limits; keep a fixed system policy separate from user "style" text; treat stored prompts as semi-trusted.
- **M-04 — Parameterized repeat count up to 5000 → input-thread DoS.** `automation_service.py` 39–41 loops `count` times on the serialized input worker; `number_parser.parse_number` default max 5000 (scroll amplifies to millions of events). CWE-400; OWASP A04. → Cap execution count (e.g. ≤20–100) independent of parse max; add per-command wall-clock budget.
- **M-05 — Destructive voice commands execute without confirmation.** `automation_command_registry.py` 26–32 (`close` → `alt+f4`), `mark_service.py` 102–104 (reset all marks). ASR false positives trigger irreversible actions; the UI path requires confirmation, the voice path does not. CWE-862; OWASP A04. → Tier commands; require a second utterance/confirmation for destructive ones.
- **M-06 — Mark-execute single-word fallback.** `text_command_parse.py` 223–228: any unmatched single token becomes `MarkExecuteCommand`, clicking saved coordinates if a label matches. CWE-863. → Require an explicit execute prefix or a dedicated mode.
- **M-07 — Sound→command mappings bypass phrase validation.** `sound_service.py` 177–183 / `parser.py` 131–135 accept any non-empty phrase (incl. parameterized) via bus/JSON, unlike the UI path. CWE-863; OWASP A01. → Validate mapped phrases against the action map.
- **M-08 — Unbounded `num_samples` in sound training.** `sound_service.py` 131–147, 208–221: event-bus-driven count is not clamped to the UI’s 1–100. CWE-400/770. → Clamp server-side; cap `_training_samples`.
- **M-09 — Per-chunk `asyncio.create_task` without coroutine backpressure.** `audio_capture_service.py` 147–151 and segmenters spawn a task per ~30 ms chunk, each retaining PCM until the bounded queue accepts it. CWE-400/770. → Single publisher coroutine with bounded buffer and drop-on-overload.
- **M-10 — Missing restrictive file/dir permissions.** `app_config.py` 971, `logging_config.py` 28/43/62, `atomic_json.py` 36–40 create dirs/files with default ACLs/umask (no `0o600`/`0o700`). CWE-276; CERT FIO06. → Set explicit permissions on data root, config, and logs.
- **M-11 — Concurrent writes to the same JSON without file-level locking.** `atomic_json.py` 34–53 + `storage_service.py` lock guards cache only; tests acknowledge races. CWE-362; CERT POS49. → Per-path lock / OS file lock around read-modify-write.
- **M-12 — Symlink-following reads in app-data tree.** `atomic_json.py` 21–25 opens paths without no-follow / containment check (TOCTOU on `path.exists()`). CWE-59/61. → Resolve and verify containment under the data root; reject non-regular files.
- **M-13 — Persisted user overrides accept out-of-range values.** `user_configurable_settings.py` 56–67 validates path names but not value ranges; many `app_config.py` fields lack `ge/le` (e.g. `context_length`, `default_rect_count`). CWE-20/400. → Add bounded `Field(ge=…, le=…)`; reject out-of-range overrides at load.
- **M-14 — Tampered aliases/prompts inject into dictation/LLM via storage.** `storage_models.py` 32–39, 82–89 (unconstrained `text`/alias values). CWE-74/353. → Length/character validation at load; treat as untrusted in the LLM/dictation layers.
- **M-15 — Supply-chain pinning gaps (CI/pre-commit/deps).** GitHub Actions pinned to floating tags (`ci-cd.yml` 18/21/31/45), pre-commit hooks pinned to tags (`.pre-commit-config.yaml`), unpinned `pip install pre-commit` in CI, loose `>=` specifiers in `pyproject.toml` (24/34) and `requirements*.txt` (incl. version-less `pip-licenses`), unpinned build backend. CWE-829/1104; OWASP A06. → Pin actions/hooks to commit SHAs; install dev tooling from the lockfile; pin or deprecate divergent requirements files; add `pip-audit`/OSV scanning.

---

### LOW

- **L-01 — Internal error details published on the event bus / shown in dialogs.** `llm_service.py` 251/267/362; UI `QMessageBox` paths. CWE-209; OWASP A09. → User-facing generic messages; detailed errors to (opt-in) logs with path redaction.
- **L-02 — Mark coordinates not validated against screen bounds.** `mark_service.py` 71–72, 250–257. CWE-1284. → Clamp to `pyautogui.size()` / per-monitor geometry.
- **L-03 — Unbounded grid click history.** `click_tracker_service.py` 144–147. CWE-400. → Ring buffer / time window; cap persisted size.
- **L-04 — Clipboard content logged on verify mismatch.** `text_input_service.py` 141–147 logs expected/actual snippets. CWE-532; OWASP A09. → Log lengths/hashes only.
- **L-05 — Latent unauthenticated grid config mutation via `setattr`.** `grid_service.py` 54–72 mutates live config from bus payloads (no current publisher). CWE-15. → Remove dead handler or route through validated config API.
- **L-06 — Repeat/grid counts up to 5000 degrade UI.** `text_command_parse.py` 140–152; `number_parser.py` 275–300. CWE-400/1284. → Separate, lower caps for grid cells vs automation repeats.
- **L-07 — `OverflowError` uncaught in float→int number parsing.** `number_parser.py` 186–190, 56–64 (`int(float("1e1000"))`). CWE-754/248. → Catch `(ValueError, OverflowError)` / `math.isfinite`.
- **L-08 — Dictation/LLM content logged at INFO.** `popup_view.py` 473/557/632/730. CWE-532; OWASP A09. → Never log transcription/LLM bodies; lengths/hashes behind a diagnostic flag only.
- **L-09 — QLabel/QMessageBox AutoText markup injection from user/OS strings.** `commands/view.py` 145–146, `labels.py`, `system_controller.py` 26–31, `llm_download_dialog.py` 109–110. CWE-79/1021. → `setTextFormat(Qt.TextFormat.PlainText)` on dynamic labels/dialogs; escape `<>&`.
- **L-10 — Unbounded pre-start event buffer + untracked UI `create_task` + leaked RPC futures + thread-per-blocking-call.** `event_bus.py` 60/137–139, `settings/controller.py` 34/79–83, `worker.py` 76–84. CWE-770/391/401. → Cap pre-start buffer; route UI async through `lifecycle.spawn()`; add future timeouts; use a bounded thread pool.
- **L-11 — Atomic write lacks `fsync` before rename.** `atomic_json.py` 39–45. CWE-404; CERT FIO32. → `flush()` + `os.fsync()` before `os.replace`.
- **L-12 — Unbounded JSON load into memory.** `atomic_json.py` 24–25. CWE-400. → Size-check before `json.load`.

---

### INFORMATIONAL

- **I-01 — `pyautogui.FAILSAFE` set only in the dictation init path** (`text_input_service.py` 33–34); set it once at app startup. CWE-754.
- **I-02 — `GridShowCommand.click_mode` is a free `str`** (`command_types.py` 304–307); use `Literal[...]`. CWE-20.
- **I-03 — Vosk/Moonshine/YAMNet models loaded from disk without signature checks** (`vosk_engine.py` 22, `sound_recognizer.py` 298–304, `moonshine_engine.py` 224–233). CWE-494. Lower risk than C-01 (not pickle), but verify bundle integrity.
- **I-04 — Substring matching for dictation stop/modifier phrases** (`command_speech_service.py` 98–107); prefer token-boundary matching. CWE-841.
- **I-05 — `ErrorResult` parse outcomes silently dropped** (`parser.py` 111–118); surface a validation event. CWE-390.
- **I-06 — `load_app_config` would honor on-disk `logging.enable_logs`** if wired in (`app_config.py` 1012–1046); currently unused. Mark security-sensitive fields non-overridable from disk before enabling. CWE-532.
- **I-07 — AST-only CI privacy guard can be bypassed** by non-literal defaults (`scripts/ci/assert_logging_disabled_by_default.py` 22–30); assert on the imported model instead. CWE-778.
- **I-08 — `MANIFEST.in` lacks `global-exclude` for secret files** (`.env`, `*.pem`); add them. CWE-200.
- **I-09 — `vocalance.py` redirects std streams to `os.devnull`** without closing handles (10–13); route to opt-in rotating log. CWE-778.

---

## 4. Cross-Cutting Themes

1. **The storage tier is the dominant trust boundary.** C-01, H-02, M-01, M-07, M-12, M-14 all stem from treating files under `%APPDATA%` as trusted. Because the app outputs keystrokes/clicks, on-disk tampering escalates to session compromise. Adopt **integrity verification (HMAC/signature) + semantic validation + safe formats (no pickle)** uniformly at the storage boundary, and fail closed.

2. **Input → OS action lacks a validation/authorization layer.** H-01, H-04, M-04, M-05, M-06 show recognized input flowing to powerful sinks (`pyautogui`, clipboard) with minimal allowlisting, bounding, or confirmation. Introduce a single chokepoint that allowlists keys, caps repeat counts, sanitizes injected text, and gates destructive actions.

3. **Supply-chain artifacts are under-verified.** H-03, H-05, H-06, M-15, I-03 — neither downloaded models, the UV bootstrap, license-tool archives, nor CI/dep pins are integrity-checked. Standardize on **hash/signature pinning** (models, actions-by-SHA, lockfile-hash installs) and verified-artifact bootstrapping.

4. **Privacy posture is strong by default but leaks when logging is on.** L-04, L-08, M-02 show dictated content (potential secrets) reaching logs/UI. Keep logging off by default (already enforced) and **never log transcription, clipboard, or LLM bodies**.

---

## 5. Suggested Remediation Order

| Phase | Items |
|-------|-------|
| **P0 — Eliminate code-exec & injection** | C-01 (drop pickle), H-01 (key allowlist), H-04 (sanitize dictation), H-05 (verified bootstrap), H-06 (safe extract) |
| **P1 — Integrity & trust boundary** | H-02, H-03, M-01, M-07, M-10, M-11, M-12, M-14 |
| **P2 — Abuse & DoS hardening** | M-02–M-06, M-08, M-09, M-13, L-02/03/06/07 |
| **P3 — Supply chain & hygiene** | M-15, I-03, I-06, I-07, I-08 |
| **P4 — Privacy, UI, lifecycle polish** | L-01, L-04, L-08, L-09, L-10, L-11, L-12, I-01/02/04/05/09 |

---

*Severity reflects a local-desktop threat model. There is no remote network attack surface; the principal risks are local tampering, adversarial/misrecognized audio turned into OS actions, the model/build supply chain, and dictation-content privacy.*
