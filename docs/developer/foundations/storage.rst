Storage and configuration
#########################

Vocalance is a local application: every piece of state it remembers
between sessions lives on the user's disk. Marks, custom commands,
trained sound mappings, click history, dictation aliases, agentic
prompts, user-configurable settings — all of it is stored in JSON
files inside the user's data directory.

This chapter describes the two services that own that state.
``StorageService`` provides typed, atomic JSON persistence with an
in-memory cache. ``RuntimeConfigurationStore`` sits on top of the
storage service and exposes the live configuration the rest of the
application reads.

Typed JSON: the storage service
===============================

``StorageService`` (``vocalance/app/services/storage/storage_service.py``)
maps each domain to a single Pydantic model and a single file.

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

Each model is a frozen-ish Pydantic ``BaseModel`` describing the
file's complete structure. Reading and writing therefore happens at
the granularity of *the whole file* — there is no partial document
update, no schema migration, no JSON Patch.

The public surface is small:

.. code-block:: python

   class StorageService:
       async def read(self, model_type: Type[StorageData]) -> StorageData: ...
       async def write(self, data: StorageData) -> bool: ...

A read returns a typed instance; a write commits one.

Atomic writes
-------------

Writing a Pydantic model to disk involves serializing it and then
replacing the contents of the destination file. A naïve "open,
truncate, write" sequence is dangerous: a crash between truncate
and write leaves the user with an empty file and lost data.

The storage service writes through ``write_json_atomic``:

1. Serialize to JSON bytes.
2. Write the bytes to a temporary file in the same directory.
3. ``os.replace`` the temp file over the destination.

``os.replace`` is an atomic rename on every supported platform.
Either the new file is in place or the old one is — never both,
never neither. A crash anywhere in the sequence leaves the file
system in a consistent state.

Cached reads
------------

Several services re-read the same file frequently. The parser asks
for the action map on every utterance; the mark service reads marks
on every command. A cold disk read every time would be wasteful and
slow.

The storage service therefore keeps an in-memory cache keyed by
model type. Each entry has a TTL (configurable, defaults to a few
seconds). A read inside the TTL returns the cached instance; a read
after the TTL re-reads from disk and refreshes the entry.

The cache is invalidated implicitly: a successful ``write`` updates
the cache entry to the new instance. Cache invalidation in the
abstract is hard, but here it is bounded — the only thing that
mutates a file is the storage service itself.

Live configuration: the runtime store
=====================================

Storage answers "what is on disk". Runtime configuration answers
"what is the application running with right now". The two are
related but distinct: a user can change a setting in the UI, the
application uses the new value immediately, and only later (or
never) is that change persisted to ``AppUserConfigDocument``.

The owner of the live configuration is ``RuntimeConfigurationStore``
(``vocalance/app/services/storage/runtime_configuration.py``). It
holds the single ``GlobalAppConfig`` instance every other service
reads from, and is responsible for keeping that instance in sync
with disk and with the UI.

Initialization
--------------

At startup, the runtime store does three things in order:

1. Read ``AppUserConfigDocument`` from disk via the storage service.
2. Sanitize the user overrides against an allow-list of permitted
   setting paths (``ALLOWED_USER_SETTING_PATHS``).
3. Apply each override to the in-memory ``GlobalAppConfig``.

The allow-list is the boundary that protects the application from
arbitrary user-supplied values: only paths that appear in the list
are accepted. Anything else is silently ignored.

Live updates
------------

When the user changes a setting through the UI, the runtime store
applies the change in two steps:

.. mermaid::

   flowchart LR
       Tab[Settings tab] -->|RuntimeConfigRequestEvent| Store[RuntimeConfigurationStore]
       Store -->|update GlobalAppConfig| Cfg[Live config]
       Store -->|publish| Bus[Event bus]
       Bus -->|SettingsChangedEvent| Sub[Subscribed services]
       Store -->|persist| Disk[AppUserConfigDocument on disk]

The two-step pattern is deliberate. Updating ``GlobalAppConfig`` is
the *immediate* change: anything reading the config the next time
sees the new value. Publishing ``SettingsChangedEvent`` lets any
service that cached a derived value re-read it; the segmenters, for
example, rebuild their threshold parameters in response.

Persistence happens in the same handler so a restart preserves the
change.

This is what allows the user to retune VAD parameters or LLM prompts
in the middle of a session and observe the effect immediately —
nothing reloads, nothing restarts, the application simply applies
the new values.

Where to read next
==================

The four foundation chapters end here. With the bus, the threading
model, the lifecycle, and the storage layer all in place, every
sentence in the *features* chapters has a concrete underpinning. The
guide ends; the codebase is the next read.
