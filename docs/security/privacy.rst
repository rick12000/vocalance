Privacy
#######

After the initial installation (if following the
`installation script <https://github.com/rick12000/vocalance/releases/latest/download/setup.ps1>`_),
the application makes no outbound network requests. All speech recognition,
command execution, dictation, and AI inference run entirely on the host machine.

While no data is ever transmitted externally, some data is stored locally on
the user's machine. This is covered in the sections below.

Stored Data
===========

Logs
----

The application has two distinct logging mechanisms: **developer logs** and an
**activity tracker**. Developer logs are conventional Python ``logging`` module
outputs. The activity tracker is a separate structured JSONL logger that
records security-salient operations (dictation and command execution).

.. admonition:: No Logs Policy by Default

   The application always ships with both logging mechanisms disabled by default. They can be individually enabled in the application config:

   .. code-block:: text

      # vocalance/app/config/logging_config.py
      enable_logs: bool = Field(default=False, ...)

      # vocalance/app/config/app_config.py → ActivityTrackingConfig
      enabled: bool = Field(default=False, ...)

   The CI pipeline enforces these defaults via static AST analysis on every
   build. A commit that flips either default to ``True`` will not produce a
   release (see :doc:`releases`).

When developer logging is enabled, output goes to stdout and to:

.. code-block:: text

   %APPDATA%\Vocalance\logs\<YYYYMMDD_HHMMSS>\app.log

Developer logs may incidentally include dictated phrases or executed command
names. They are intended for active development only.

When the activity tracker is enabled, it writes one JSON record per event to:

.. code-block:: text

   %APPDATA%\Vocalance\activity_logs\activity_<YYYYMMDD_HHMMSS>.jsonl

Each record follows this structure:

.. code-block:: json

   {
     "timestamp": "2026-06-21T00:30:00.000000+00:00",
     "run_id": "<uuid4>",
     "event_type": "dictation",
     "dictation": {
       "text": "the final typed text",
       "mode": "standard",
       "session_id": "<uuid4>",
       "llm_enhanced": false,
       "active_modifiers": []
     }
   }

.. code-block:: json

   {
     "timestamp": "2026-06-21T00:30:05.000000+00:00",
     "run_id": "<uuid4>",
     "event_type": "automation",
     "automation": {
       "command_key": "copy",
       "action_type": "hotkey",
       "action_value": "ctrl+c",
       "count": 1,
       "is_custom": false,
       "functional_group": "clipboard",
       "short_description": "Copy selection"
     }
   }

The activity tracker is intended for enterprise deployments requiring an audit
trail of OS-level commands and dictation output. Treat these files as sensitive
when enabled.

User Data
---------

Besides logs, the application stores and persists a range of user-specific data. This may include:

- user settings (integer or boolean values controlling preferences over VAD sensitivity, grid cell count, etc.)
- JSON mappings for custom command hotkeys and command phrase overrides
- user generated sound recordings (if the user has recorded any)
- named screen position coordinates to support mark functionality
- saved LLM prompts (if the AI feature is enabled and the user has provided any)
- dictation aliases (if the user has defined any)

No dictation outputs or user audio is ever stored on disk.

All of this data is stored and freely auditable by the user at:

.. code-block:: text

   %APPDATA%\Vocalance\

.. admonition:: Data Classification

   None of the above data is intrinsically sensitive. Three categories could contain
   sensitive content depending on what the user puts in them:

   - **Sound recordings** — audio samples used for custom sound commands. These
     are typically short non-verbal clips, though the user controls what is
     recorded.
   - **Saved LLM prompts** — prompt templates for the Smart and Amend dictation
     modes. These are only present if the AI functionality was enabled at
     installation time. Their sensitivity depends entirely on what the user
     writes in them.
   - **Dictation Aliases** — user-defined text that can be substituted for other text during dictation.
     See :ref:`security/input_validation:Alias Sanitisation` for the validation controls applied to alias content.
