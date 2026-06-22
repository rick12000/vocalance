Input Validation
#################

Vocalance validates user input at two layers: at the point of entry (UI or
service layer) and again on every ingestion from disk.

Settings Bounds
===============

User-adjustable settings are declared as Pydantic fields with explicit
``ge``/``le`` constraints:

.. code-block:: python

   class LLMConfig(BaseModel):
       context_length: int = Field(default=2048, ge=512,  le=32768)
       max_tokens:     int = Field(default=512,  ge=64,   le=4096)

   class GridConfig(BaseModel):
       default_rect_count: int = Field(default=9, ge=4, le=400)

Values outside these ranges are rejected by Pydantic at deserialisation time.

Which settings are user-writable is separately controlled by an allowlist of
dot-path keys in
`vocalance/app/services/storage/user_configurable_settings.py <https://github.com/rick12000/vocalance/blob/main/vocalance/app/services/storage/user_configurable_settings.py>`_:

.. code-block:: python

   ALLOWED_USER_SETTING_PATHS: FrozenSet[str] = frozenset({
       "llm.context_length",
       "llm.max_tokens",
       "llm.selected_model_id",
       "grid.default_rect_count",
       "sound_recognizer.sound_threshold",
       "sound_recognizer.ambient_threshold",
       "vad.command_silent_chunks_for_end",
   })

Any key in ``AppUserConfigDocument.overrides`` not present in this set is
silently discarded before the config is applied to the live ``GlobalAppConfig``.
Every update is additionally dry-run validated (``model_copy(deep=True)`` +
apply) before being persisted — a value that passes the allowlist but fails a
Pydantic constraint is still rejected.

Ingestion Guards
================

Custom commands and sound mappings are marked security-sensitive at the storage
layer. These guards form part of the tampered local storage defences described in
:ref:`security/security_assumptions:Tampered Local Storage`.

.. code-block:: python

   _SECURITY_SENSITIVE_MODELS: FrozenSet[Type[StorageData]] = frozenset({
       CommandsData,
       SoundMappingsData,
   })

A ``ValidationError`` or read error on either model causes the storage service
to return an empty safe default, record the file as corrupt, and publish a
``StorageCorruptionWarningEvent``. The UI surfaces a modal offering to delete
the corrupt file. A tampered or malformed file causes a safe reset — not a crash
and not partial execution.

Hotkey Allowlist
================

Custom commands map a spoken phrase to a hotkey. This is the highest-risk
user-configurable input: an unvalidated value could trigger arbitrary key
sequences or macro chains.

Validation runs at three independent points — UI submission, service-layer
persistence, and storage ingestion on every load — using a single function in
`vocalance/app/config/hotkey_validation.py <https://github.com/rick12000/vocalance/blob/main/vocalance/app/config/hotkey_validation.py>`_:

.. code-block:: python

   def is_valid_custom_hotkey(value: str) -> bool:
       if not value or not value.strip():
           return False
       if any(c in value for c in (",", ";")):
           return False
       tokens = [t for t in value.replace(" ", "+").split("+") if t]
       return bool(tokens) and all(is_allowed_key(t) for t in tokens)

   def is_allowed_key(token: str) -> bool:
       candidate = token.strip().lower()
       return bool(
           candidate and (
               (len(candidate) == 1 and (candidate.isalnum() or candidate in SYMBOL_KEYS))
               or candidate in MODIFIER_KEYS
               or candidate in NAMED_KEYS
               or FUNCTION_KEY_RE.fullmatch(candidate)
           )
       )

Two properties are enforced:

- **No sequences.** ``,`` and ``;`` are rejected outright — they could express
  multiple sequential actions (e.g. ``ctrl+c, ctrl+v``). A custom command can
  only express a single simultaneous combination.
- **Token allowlist.** Each ``+``-separated token must be a modifier
  (``ctrl``, ``alt``, ``shift``, ``win``), a named key (``enter``, ``tab``,
  arrow keys, etc.), a single alphanumeric or symbol character, or ``f1``–``f24``.

The underlying assumption is that a single simultaneous combination, however
many keys it includes, has a bounded and predictable effect. A sequence does not.

Any command on disk that fails this check is dropped at load time and never
reaches the automation service:

.. code-block:: python

   @model_validator(mode="after")
   def filter_invalid_custom_commands(self) -> "CommandsData":
       valid: Dict[str, AutomationCommand] = {}
       for phrase, cmd in self.custom_commands.items():
           if cmd.action_type == "hotkey" and is_valid_custom_hotkey(cmd.action_value):
               valid[phrase] = cmd
           else:
               logger.warning(
                   "Security: dropping command %r on load — "
                   "action_type=%r action_value=%r failed validation",
                   phrase, cmd.action_type, cmd.action_value,
               )
       self.custom_commands = valid
       return self

Commands with any ``action_type`` other than ``"hotkey"`` are also dropped. The
UI only creates hotkey commands; the storage layer enforces this independently.

Alias Sanitisation
==================

Aliases substitute a spoken trigger with text typed via the clipboard. An alias
value containing a newline, carriage return, or Unicode line-break equivalent
could silently execute its preceding text if pasted into a terminal.

All alias keys and values are validated against
`vocalance/app/config/alias_validation.py <https://github.com/rick12000/vocalance/blob/main/vocalance/app/config/alias_validation.py>`_:

.. code-block:: python

   DANGEROUS_CHAR_RE = re.compile(r"[\x00-\x1f\x7f\u0085\u2028\u2029]")

   def is_valid_alias_text(value: str) -> bool:
       return not DANGEROUS_CHAR_RE.search(value)

This blocks all ASCII control characters (``\x00``–``\x1f``: newline,
carriage return, tab, escape, and others), DEL (``\x7f``), Unicode NEL
(``\u0085``), and the Unicode line and paragraph separators (``\u2028``,
``\u2029``) — all of which many parsers treat as line terminators.

At ingestion, ``DictationAliasData`` enforces both the control-character check
and length limits (100 chars for keys, 2000 for values):

.. code-block:: python

   @field_validator("aliases")
   @classmethod
   def validate_aliases(cls, v: Dict[str, str]) -> Dict[str, str]:
       for key, value in v.items():
           if len(key) > MAX_ALIAS_KEY_LENGTH:
               raise ValueError(...)
           if len(value) > MAX_ALIAS_VALUE_LENGTH:
               raise ValueError(...)
           if not is_valid_alias_text(key) or not is_valid_alias_text(value):
               raise ValueError(f"Alias '{key}' contains characters that are not permitted")
       return v

A failing alias raises ``ValidationError``; the storage service returns empty
defaults. Alias output is always inert printable text — it cannot trigger
execution in any context that treats control characters as line terminators.
