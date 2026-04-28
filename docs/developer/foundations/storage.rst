Storage and configuration
#########################

Vocalance is a local application: every piece of state it
remembers between sessions lives on the user's disk. Marks,
custom commands, trained sound mappings, click history, dictation
aliases, agentic prompts, user settings — all of it is stored as
JSON files in the user's data directory.

Two services own that state. ``StorageService`` provides typed,
atomic JSON persistence with an in-memory cache.
``RuntimeConfigurationStore`` sits on top of it and exposes the
live configuration the rest of the application reads.

Layer at a glance
=================

.. mermaid::

   flowchart LR
       Svcs[Services] -->|read/write typed model| Storage[StorageService]
       Storage -->|atomic write,<br/>cached read| Disk[(JSON files)]
       Storage -.boots from.-> Runtime[RuntimeConfigurationStore]
       UI[Settings tab] -->|RuntimeConfigRequestEvent| Runtime
       Runtime -->|update GlobalAppConfig| Cfg[Live config<br/><i>read by all services</i>]
       Runtime -->|persist override| Storage

Every service that needs persistence talks to ``StorageService``;
nothing reads or writes JSON directly. Everything that needs
*current* configuration reads from ``GlobalAppConfig``, which the
runtime store keeps in sync with disk and with the UI.

Typed JSON: StorageService
==========================

``StorageService``
(``vocalance/app/services/storage/storage_service.py``) maps each
domain to one Pydantic model and one file.

==========================  ================================================
Model                       What it holds
==========================  ================================================
``MarksData``               Saved marks (label → coordinate).
``CommandsData``            User-configured automations.
``GridClicksData``          Grid click history (for re-ranking).
``AgenticPromptsData``      LLM rewrite prompts.
``SoundMappingsData``       Sound label → command phrase.
``DictationAliasData``      Dictation alias expansions.
``AppUserConfigDocument``   User-settable configuration overrides.
==========================  ================================================

Each model is a Pydantic ``BaseModel`` describing the file's
complete structure. Reading and writing happen at the granularity
of *the whole file* — there is no partial document update, no
schema migration, no JSON Patch.

The public surface is small:

.. code-block:: python

   class StorageService:
       async def read(self, model_type: Type[StorageData]) -> StorageData: ...
       async def write(self, data: StorageData) -> bool: ...

A read returns a typed instance; a write commits one.

Atomic writes
-------------

A naïve "open, truncate, write" sequence is dangerous: a crash
between truncate and write leaves an empty file. The storage
service writes through ``write_json_atomic``.

.. mermaid::

   flowchart LR
       Model[Pydantic model] --> Bytes[Serialize<br/>to JSON bytes]
       Bytes --> Tmp[Write bytes<br/>to temp file<br/><i>same directory</i>]
       Tmp --> Repl[os.replace<br/>temp → destination]
       Repl --> Done[(Final file)]

``os.replace`` is an atomic rename on every supported platform.
Either the new file is in place or the old one is — never both,
never neither. A crash anywhere in the sequence leaves the
filesystem in a consistent state.

Cached reads
------------

Several services re-read the same file frequently. The parser
asks for the action map on every utterance; the mark service
reads marks on every command. A cold disk read every time would
be wasteful.

The storage service keeps an in-memory cache keyed by model
type, with a per-entry TTL.

================  ============================================================
Read happens      Effect
================  ============================================================
Within TTL        Return cached instance.
After TTL         Re-read from disk, refresh entry, return.
After ``write``   Cache entry is replaced with the just-written instance.
================  ============================================================

Cache invalidation in the abstract is hard; here it is bounded
because the only thing that mutates a file is the storage
service itself.

Live configuration: RuntimeConfigurationStore
=============================================

Storage answers "what is on disk". Runtime configuration answers
"what is the application running with right now". A user can
change a setting in the UI, the application uses the new value
immediately, and only later (or never) is that change persisted
to ``AppUserConfigDocument``.

The owner of the live configuration is
``RuntimeConfigurationStore``
(``vocalance/app/services/storage/runtime_configuration.py``). It
holds the single ``GlobalAppConfig`` instance every other service
reads from.

Initialization
--------------

At startup, the runtime store does three things in order.

#. Read ``AppUserConfigDocument`` from disk via the storage
   service.
#. Sanitize the user overrides against an allow-list of
   permitted setting paths (``ALLOWED_USER_SETTING_PATHS``).
#. Apply each override to the in-memory ``GlobalAppConfig``.

The allow-list is the boundary that protects the application
from arbitrary user-supplied values: only paths in the list are
accepted. Anything else is silently ignored.

Live updates
------------

When the user changes a setting, the runtime store applies the
change in two parallel streams.

.. mermaid::

   flowchart LR
       Tab[Settings tab] -->|RuntimeConfigRequestEvent| Store[RuntimeConfigurationStore]
       Store --> Mem[Update in-memory<br/>GlobalAppConfig]
       Store --> Notify[Publish<br/>SettingsChangedEvent]
       Store --> Persist[Persist to<br/>AppUserConfigDocument]
       Notify -->|deliver| Subs[Subscribed services<br/><i>e.g. segmenters re-tune VAD</i>]

The two-step pattern is deliberate. Updating ``GlobalAppConfig``
is the *immediate* change: anything reading the config the next
time sees the new value. Publishing ``SettingsChangedEvent``
lets services that cached a derived value re-read it; the
segmenters, for example, rebuild their threshold parameters in
response. Persistence happens in the same handler so a restart
preserves the change.

This is what allows the user to retune VAD parameters or LLM
prompts mid-session and observe the effect immediately —
nothing reloads, nothing restarts, the application simply
applies the new values.

Where to read next
==================

The four foundation chapters end here. With the bus, the
threading model, the lifecycle, and the storage layer all in
place, every sentence in the *features* chapters has a concrete
underpinning. The guide ends; the codebase is the next read.
