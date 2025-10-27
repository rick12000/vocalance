API Reference
=============

This section provides detailed documentation for Vocalance's main classes and modules.

Core Services
=============

.. currentmodule:: vocalance.app.services

Speech Recognition
------------------

.. currentmodule:: vocalance.app.services.audio.stt.stt_service

SpeechToTextService
~~~~~~~~~~~~~~~~~~~
.. autoclass:: SpeechToTextService
   :members:
   :exclude-members: __init__
   :noindex:

.. currentmodule:: vocalance.app.services.audio.simple_audio_service

SimpleAudioService
~~~~~~~~~~~~~~~~~~
.. autoclass:: SimpleAudioService
   :members:
   :exclude-members: __init__
   :noindex:

Command Processing
------------------

.. currentmodule:: vocalance.app.services.centralized_command_parser

CentralizedCommandParser
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: CentralizedCommandParser
   :members:
   :exclude-members: __init__
   :noindex:

.. currentmodule:: vocalance.app.services.automation_service

AutomationService
~~~~~~~~~~~~~~~~~
.. autoclass:: AutomationService
   :members:
   :exclude-members: __init__
   :noindex:

Grid and Screen Interaction
---------------------------

.. currentmodule:: vocalance.app.services.grid.grid_service

GridService
~~~~~~~~~~~
.. autoclass:: GridService
   :members:
   :exclude-members: __init__
   :noindex:

.. currentmodule:: vocalance.app.services.mark_service

MarkService
~~~~~~~~~~~
.. autoclass:: MarkService
   :members:
   :exclude-members: __init__
   :noindex:

Dictation and AI
-----------------

.. currentmodule:: vocalance.app.services.audio.dictation_handling.dictation_coordinator

DictationCoordinator
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: DictationCoordinator
   :members:
   :exclude-members: __init__
   :noindex:

Storage and Configuration
-------------------------

.. currentmodule:: vocalance.app.services.storage.storage_service

StorageService
~~~~~~~~~~~~~~
.. autoclass:: StorageService
   :members:
   :exclude-members: __init__
   :noindex:

.. currentmodule:: vocalance.app.services.storage.settings_service

SettingsService
~~~~~~~~~~~~~~~
.. autoclass:: SettingsService
   :members:
   :exclude-members: __init__
   :noindex:

.. currentmodule:: vocalance.app.services.storage.settings_update_coordinator

SettingsUpdateCoordinator
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: SettingsUpdateCoordinator
   :members:
   :exclude-members: __init__
   :noindex:

Sound Recognition
-----------------

.. currentmodule:: vocalance.app.services.audio.sound_recognizer.streamlined_sound_service

StreamlinedSoundService
~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: StreamlinedSoundService
   :members:
   :exclude-members: __init__
   :noindex:

User Interface
==============

.. currentmodule:: vocalance.app.ui

Main Application Window
-----------------------

.. currentmodule:: vocalance.app.ui.main_window

AppControlRoom
~~~~~~~~~~~~~~
.. autoclass:: AppControlRoom
   :members:
   :exclude-members: __init__
   :noindex:

Startup and Initialization
---------------------------

.. currentmodule:: vocalance.app.ui.startup_window

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

.. currentmodule:: vocalance.app.config

Application Configuration
-------------------------

.. currentmodule:: vocalance.app.config.app_config

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

.. currentmodule:: vocalance.app.event_bus

EventBus
~~~~~~~~
.. autoclass:: EventBus
   :members:
   :exclude-members: __init__
   :noindex:

Utility Functions
=================

.. currentmodule:: vocalance.app.ui.utils

UI Thread Management
--------------------

.. currentmodule:: vocalance.app.ui.utils.ui_thread_utils

.. autofunction:: initialize_ui_scheduler

Icon Management
---------------

.. currentmodule:: vocalance.app.ui.utils.ui_icon_utils

.. autofunction:: set_window_icon_robust

Configuration Loading
---------------------

.. currentmodule:: vocalance.app.config.app_config

.. autofunction:: load_app_config

Logging Configuration
---------------------

.. currentmodule:: vocalance.app.config.logging_config

.. autofunction:: setup_logging

Data Models and Enums
=====================

STT Modes
---------

.. currentmodule:: vocalance.app.services.audio.stt.stt_service

.. autoclass:: STTMode
   :members:
   :noindex:

Storage Data Models
--------------------

.. currentmodule:: vocalance.app.services.storage.storage_models

.. autoclass:: StorageData
   :members:
   :noindex:

.. autoclass:: SettingsData
   :members:
   :noindex:

.. autoclass:: CommandsData
   :members:
   :noindex:

.. autoclass:: MarksData
   :members:
   :noindex:
