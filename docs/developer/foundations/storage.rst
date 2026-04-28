Storage
#######

.. sectnum::

Services read configuration, marks, automations, and other state from disk.
:doc:`lifecycle` described how services are constructed and torn down; this
chapter describes how they persist and reload state across those restarts.

All state Vocalance retains between sessions lives in JSON files on disk. Every
read and write goes through a single service. On top of that raw persistence
layer sits a live configuration store that keeps the in-memory application
configuration in sync with disk and with the user's current UI settings.

The Two Layers
===============

The storage layer has two distinct responsibilities that are kept in separate
services.

.. mermaid::

   flowchart LR
       subgraph Persistence["Persistence Layer"]
           Storage[StorageService]
           Disk[(JSON files)]
           Storage -->|atomic write, cached read| Disk
       end
       subgraph Config["Live Configuration"]
           Runtime[RuntimeConfigurationStore]
           Cfg[GlobalAppConfig]
           Runtime -->|update| Cfg
       end
       Svcs[Services] -->|read/write typed model| Storage
       Storage -.boots from.-> Runtime
       UI[Settings tab] -->|RuntimeConfigRequestEvent| Runtime
       Runtime -->|persist override| Storage

**``StorageService``** owns the mapping from domain concepts (marks, sounds,
automations, aliases) to files on disk. It provides typed reads and writes,
atomic file replacement, and an in-memory cache. It is not responsible for
keeping any in-memory data structure up to date — it is purely a persistence
adapter.

**``RuntimeConfigurationStore``** owns the live ``GlobalAppConfig`` instance
that every service reads for its current operating parameters. It uses
``StorageService`` to read and write ``AppUserConfigDocument``, applies the
stored overrides to the in-memory config at startup, and keeps both the
in-memory config and the disk file in sync when the user changes a setting.

Raw Persistence: StorageService
=================================

Typed Models and Files
-----------------------

``StorageService``
(``vocalance/app/services/storage/storage_service.py``) maps each domain to a
single Pydantic ``BaseModel`` subclass and a single JSON file. The public
interface is two coroutines:

.. code-block:: python

   class StorageService:
       async def read(self, model_type: Type[StorageData]) -> StorageData: ...
       async def write(self, data: StorageData) -> bool: ...

Passing a model type to ``read`` returns a fully validated instance of that
model. Passing an instance to ``write`` serialises it and commits it to the
corresponding file. There is no partial update, no query language, no schema
migration — each file is always written and read as a complete document.

The domain models and their files:

.. list-table::
   :widths: 35 65
   :header-rows: 1
   :class: uniform-rows

   * - Model
     - Contents
   * - ``MarksData``
     - Saved mark labels and their screen coordinates.
   * - ``CommandsData``
     - User-configured voice automation actions.
   * - ``GridClicksData``
     - Click frequency data used to rank grid cells.
   * - ``AgenticPromptsData``
     - LLM rewrite prompt templates for Smart and Amend modes.
   * - ``SoundMappingsData``
     - Sound label to command phrase mappings.
   * - ``DictationAliasData``
     - Alias trigger phrases and their expanded text.
   * - ``AppUserConfigDocument``
     - User-applied overrides to the default application configuration.

Atomic Writes
--------------

Overwriting a file in place is dangerous: a crash or power loss after the
original content is erased but before the new content is fully written leaves
an empty or truncated file. The storage service avoids this by never overwriting
in place.

All writes go through ``write_json_atomic``, which uses a write-then-rename
strategy:

.. mermaid::

   flowchart LR
       Model[Pydantic model] --> Bytes[Serialise to JSON bytes]
       Bytes --> Tmp[Write to temp file<br/>same directory as target]
       Tmp --> Rename[os.replace temp → target]
       Rename --> Done[(File on disk)]

The key step is ``os.replace``. On POSIX systems this is an atomic directory
operation: the directory entry for the target path is updated to point to the
new inode in a single kernel call. On Windows, ``os.replace`` is not guaranteed
atomic at the kernel level, but it is as close as Python can get without using
transactional NTFS. Either the rename completes and the new file is visible, or
it does not and the original file is untouched. A crash at any point in the
sequence leaves the filesystem in a known consistent state.

Cached Reads
-------------

Several services read their domain file on every command. The mark service reads
mark data on every mark-execute; the parser reads the automation map on every
input. A full disk read each time would add milliseconds to every command's
latency.

``StorageService`` maintains a per-model-type in-memory cache with a
configurable TTL (time-to-live). The cache is keyed by model type, so each
domain has its own entry.

.. list-table::
   :widths: 30 70
   :header-rows: 1
   :class: uniform-rows

   * - Read condition
     - What happens
   * - Cache entry exists and is within TTL
     - Return the cached Pydantic instance immediately.
   * - Cache entry is missing or has expired
     - Read from disk, deserialise, store in cache with a fresh TTL,
       and return the new instance.
   * - A ``write`` just completed
     - Replace the cache entry with the just-written instance
       immediately, regardless of the TTL. The cache is always
       consistent with what was last written.

The cache invalidation policy is simple and correct because the storage service
is the *only writer*. No other component writes to these files while the
application is running; the cache can never be stale relative to an external
modification.

Live Configuration: RuntimeConfigurationStore
==============================================

What "Live Configuration" Means
---------------------------------

``StorageService`` answers "what is currently on disk". That is not always
what the application should be using right now. A user can change a VAD
sensitivity slider mid-session and expect the new value to take effect
immediately, before any disk write has occurred. The application's running
parameters and the on-disk representation can temporarily diverge.

``RuntimeConfigurationStore``
(``vocalance/app/services/storage/runtime_configuration.py``) bridges this gap.
It owns a single ``GlobalAppConfig`` instance — the authoritative source for
every parameter that changes the application's behaviour. Every service that
needs a configuration value reads it from ``GlobalAppConfig``, never from disk
directly.

Startup: Loading Stored Overrides
-----------------------------------

At initialization, the runtime store bootstraps ``GlobalAppConfig`` in three
steps:

1. Read ``AppUserConfigDocument`` from disk via ``StorageService``.
2. Filter the loaded overrides through ``ALLOWED_USER_SETTING_PATHS`` — a
   hardcoded allowlist of configuration keys the user is permitted to change.
   Any key not in the allowlist is silently ignored, preventing arbitrary
   values from reaching internal parameters.
3. Apply each permitted override to the in-memory ``GlobalAppConfig`` by
   traversing the key path and setting the value.

After this, ``GlobalAppConfig`` reflects exactly what the user had configured
at the end of the previous session.

Live Updates During a Session
-------------------------------

When the user changes a setting in the Settings tab, the controller publishes
``RuntimeConfigRequestEvent`` carrying the changed key and new value. The
runtime store handles this event in three parallel actions:

.. mermaid::

   flowchart LR
       Tab[Settings tab] -->|RuntimeConfigRequestEvent| Store[RuntimeConfigurationStore]
       Store --> Mem[1. Update GlobalAppConfig immediately]
       Store --> Notify[2. Publish SettingsChangedEvent]
       Store --> Persist[3. Persist to AppUserConfigDocument]
       Notify -->|deliver| Subs[Subscribed services]

**Action 1** takes effect immediately: any service that reads from
``GlobalAppConfig`` on its next operation sees the new value. No restart, no
reload.

**Action 2** publishes ``SettingsChangedEvent``. This event exists for services
that have cached derived values and need to react to a configuration change
explicitly. For example, the segmenters cache their computed energy threshold
(noise-floor estimate × multiplier). When ``SettingsChangedEvent`` arrives, they
recompute the threshold from the new ``GlobalAppConfig`` value. Without this
event, a service that cached a derived value at construction time would never
notice the change.

**Action 3** persists the override to ``AppUserConfigDocument`` via
``StorageService``. Because this runs in the same handler as action 1, the
change is durable as soon as the handler completes. A restart immediately after
a settings change will load the new value from disk and apply it.

The net effect is that a user adjusting VAD sensitivity sees the recognition
behaviour change on the next phrase, without any restart or mode switch, and
the change survives a restart.
