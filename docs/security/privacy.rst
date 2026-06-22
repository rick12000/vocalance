Privacy
#######

.. sectnum::

After the initial installation (if following the
`setup script <https://github.com/rick12000/vocalance/releases/latest/download/setup.ps1>`_),
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
output — trace-level output covering service events, recognition results, errors,
and tracebacks. The activity tracker is a separate structured JSONL logger that
records security-salient operations: every dictation output and every automation
executed.

.. admonition:: Both logging mechanisms are disabled by default

   They can be individually enabled in the application config:

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

   %LOCALAPPDATA%\vocalance_voice_assistant\cache\logs\<YYYYMMDD_HHMMSS>\app.log

Developer logs may incidentally include dictated phrases or executed command
names. They are intended for active development only.

When the activity tracker is enabled, it writes one JSON record per event to:

.. code-block:: text

   %APPDATA%\vocalance_voice_assistant_data\activity_logs\activity_<YYYYMMDD_HHMMSS>.jsonl

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

Across sessions, the application persists user configuration as JSON files under:

.. code-block:: text

   %APPDATA%\vocalance_voice_assistant_data\

This includes: application settings (VAD sensitivity, grid cell count, LLM token
limits, selected model), custom voice-to-hotkey command mappings, dictation
aliases, named on-screen mark positions, sound-to-command mappings, and saved
LLM prompt templates (if the AI feature is enabled).

.. admonition:: Sensitivity

   None of the above is intrinsically sensitive. Two fields could contain
   sensitive content depending on what the user puts in them:

   - **Sound recordings** — audio samples used for custom sound commands. These
     are typically short non-verbal clips, though the user controls what is
     recorded.
   - **Saved LLM prompts** — prompt templates for the Smart and Amend dictation
     modes. These are only present if the AI functionality was enabled at
     installation time. Their sensitivity depends entirely on what the user
     writes in them.

   No dictation output is ever stored. No background audio is ever persisted
   beyond the sound-mapping samples the user explicitly records.
