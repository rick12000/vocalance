API Reference
=============

This section provides detailed documentation for Iris's main classes and modules.

Core Services
=============

.. currentmodule:: iris.app.services

Speech Recognition
------------------

.. currentmodule:: iris.app.services.audio.stt_service

SpeechToTextService
~~~~~~~~~~~~~~~~~~~
.. autoclass:: SpeechToTextService
   :members:
   :exclude-members: __init__
   :noindex:

.. currentmodule:: iris.app.services.audio.simple_audio_service

SimpleAudioService
~~~~~~~~~~~~~~~~~~
.. autoclass:: SimpleAudioService
   :members:
   :exclude-members: __init__
   :noindex:

Command Processing
------------------

.. currentmodule:: iris.app.services.centralized_command_parser

CentralizedCommandParser
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: CentralizedCommandParser
   :members:
   :exclude-members: __init__
   :noindex:

.. currentmodule:: iris.app.services.automation_service

AutomationService
~~~~~~~~~~~~~~~~~
.. autoclass:: AutomationService
   :members:
   :exclude-members: __init__
   :noindex:

Grid and Screen Interaction
---------------------------

.. currentmodule:: iris.app.services.grid.grid_service

GridService
~~~~~~~~~~~
.. autoclass:: GridService
   :members:
   :exclude-members: __init__
   :noindex:

.. currentmodule:: iris.app.services.mark_service

MarkService
~~~~~~~~~~~
.. autoclass:: MarkService
   :members:
   :exclude-members: __init__
   :noindex:

Dictation and AI
-----------------

.. currentmodule:: iris.app.services.audio.dictation_handling.dictation_coordinator

DictationCoordinator
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: DictationCoordinator
   :members:
   :exclude-members: __init__
   :noindex:

Storage and Configuration
-------------------------

.. currentmodule:: iris.app.services.storage.unified_storage_service

UnifiedStorageService
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: UnifiedStorageService
   :members:
   :exclude-members: __init__
   :noindex:

.. currentmodule:: iris.app.services.storage.storage_adapters

StorageAdapterFactory
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: StorageAdapterFactory
   :members:
   :exclude-members: __init__
   :noindex:

Sound Recognition
-----------------

.. currentmodule:: iris.app.services.audio.sound_recognizer.streamlined_sound_service

StreamlinedSoundService
~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: StreamlinedSoundService
   :members:
   :exclude-members: __init__
   :noindex:

User Interface
==============

.. currentmodule:: iris.app.ui

Main Application Window
-----------------------

.. currentmodule:: iris.app.ui.main_window

AppControlRoom
~~~~~~~~~~~~~~
.. autoclass:: AppControlRoom
   :members:
   :exclude-members: __init__
   :noindex:

Startup and Initialization
---------------------------

.. currentmodule:: iris.app.ui.startup_window

StartupWindow
~~~~~~~~~~~~~
.. autoclass:: StartupWindow
   :members:
   :exclude-members: __init__
   :noindex:

StartupProgressTracker
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: StartupProgressTracker
   :members:
   :exclude-members: __init__
   :noindex:

Configuration and Events
========================

.. currentmodule:: iris.app.config

Application Configuration
-------------------------

.. currentmodule:: iris.app.config.app_config

GlobalAppConfig
~~~~~~~~~~~~~~~
.. autoclass:: GlobalAppConfig
   :members:
   :exclude-members: __init__
   :noindex:

AppInfoConfig
~~~~~~~~~~~~~
.. autoclass:: AppInfoConfig
   :members:
   :exclude-members: __init__
   :noindex:

Event System
------------

.. currentmodule:: iris.app.event_bus

EventBus
~~~~~~~~
.. autoclass:: EventBus
   :members:
   :exclude-members: __init__
   :noindex:

Utility Functions
=================

.. currentmodule:: iris.app.ui.utils

UI Thread Management
--------------------

.. currentmodule:: iris.app.ui.utils.ui_thread_utils

.. autofunction:: initialize_ui_scheduler

Icon Management
---------------

.. currentmodule:: iris.app.ui.utils.ui_icon_utils

.. autofunction:: set_window_icon_robust

Configuration Loading
---------------------

.. currentmodule:: iris.app.config.app_config

.. autofunction:: load_app_config

Logging Configuration
---------------------

.. currentmodule:: iris.app.config.logging_config

.. autofunction:: setup_logging

Data Models and Enums
=====================

STT Modes
---------

.. currentmodule:: iris.app.services.audio.stt_service

.. autoclass:: STTMode
   :members:
   :noindex:

Storage Types
-------------

.. currentmodule:: iris.app.services.storage.unified_storage_service

.. autoclass:: StorageType
   :members:
   :noindex:

.. autoclass:: StorageKey
   :members:
   :noindex: