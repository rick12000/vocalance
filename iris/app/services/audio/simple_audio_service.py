"""
Streamlined Audio Service

Manages dual independent audio recorders for command and dictation modes.
Provides clear separation of concerns and optimized threading.
"""

import logging
import threading
import asyncio
import time
from typing import Optional

from iris.app.event_bus import EventBus
from iris.app.services.audio.recorder import AudioRecorder

from iris.app.events.core_events import (
    CommandAudioSegmentReadyEvent,
    DictationAudioSegmentReadyEvent,
    RecordingTriggerEvent,
    AudioDetectedEvent
)
from iris.app.events.base_event import BaseEvent
from iris.app.events.dictation_events import DictationModeDisableOthersEvent, AudioModeChangeRequestEvent
from iris.app.config.app_config import GlobalAppConfig


logger = logging.getLogger(__name__)

class SimpleAudioService:
    """Streamlined audio service with independent dual recorders"""
    
    def __init__(self, event_bus: EventBus, config: GlobalAppConfig, main_event_loop=None):
        """
        Initialize the streamlined audio service
        
        Args:
            event_bus: The event bus for publishing/subscribing to events
            config: The global application configuration
            main_event_loop: The main asyncio event loop for scheduling async tasks
        """
        self._event_bus = event_bus
        self._config = config
        self._main_event_loop = main_event_loop or asyncio.get_event_loop()
        
        # Mode tracking
        self._is_dictation_mode = False
        self._lock = threading.Lock()
        
        # Independent recorders
        self._command_recorder = None
        self._dictation_recorder = None
        
        # Initialize recorders
        self._initialize_recorders()
        
        logger.info("SimpleAudioService initialized with dual independent recorders")

    def _initialize_recorders(self):
        """Initialize both command and dictation recorders"""
        try:
            # Command recorder - optimized for speed
            self._command_recorder = AudioRecorder(
                app_config=self._config,
                mode="command",
                on_audio_segment=self._on_command_audio_segment,
                on_audio_detected=self._on_audio_detected
            )
            
            # Dictation recorder - optimized for accuracy
            self._dictation_recorder = AudioRecorder(
                app_config=self._config,
                mode="dictation", 
                on_audio_segment=self._on_dictation_audio_segment
            )
            
            logger.info("Dual audio recorders initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize audio recorders: {e}", exc_info=True)
            raise

    def _on_command_audio_segment(self, segment_bytes: bytes):
        """Handle command mode audio segments - optimized for speed"""
        try:
            event = CommandAudioSegmentReadyEvent(
                audio_bytes=segment_bytes,
                sample_rate=self._config.audio.sample_rate
            )
            self._publish_audio_event(event)
            logger.debug(f"Command audio: {len(segment_bytes)} bytes")
        except Exception as e:
            logger.error(f"Error handling command audio: {e}")
    
    def _on_audio_detected(self):
        """Handle audio detection event for Markov fast-track"""
        try:
            event = AudioDetectedEvent(timestamp=time.time())
            self._publish_audio_event(event)
        except Exception as e:
            logger.error(f"Error publishing audio detected event: {e}")

    def _on_dictation_audio_segment(self, segment_bytes: bytes):
        """Handle dictation mode audio segments - optimized for accuracy"""
        try:
            logger.info(f"Publishing dictation audio segment: {len(segment_bytes)} bytes at {self._config.audio.sample_rate}Hz")
            event = DictationAudioSegmentReadyEvent(
                audio_bytes=segment_bytes,
                sample_rate=self._config.audio.sample_rate
            )
            self._publish_audio_event(event)
            logger.debug(f"Dictation audio: {len(segment_bytes)} bytes")
        except Exception as e:
            logger.error(f"Error handling dictation audio: {e}", exc_info=True)

    def _on_audio_detected(self):
        """Handle audio detection callback from command recorder"""
        try:
            event = AudioDetectedEvent(timestamp=time.time())
            self._publish_audio_event(event)
        except Exception as e:
            logger.error(f"Error handling audio detected: {e}")

    def _publish_audio_event(self, event_data: BaseEvent):
        """Unified event publication method"""
        if self._main_event_loop and not self._main_event_loop.is_closed():
            asyncio.run_coroutine_threadsafe(
                self._event_bus.publish(event_data),
                self._main_event_loop
            )

    def init_listeners(self):
        """Subscribe to relevant events."""
        self._event_bus.subscribe(event_type=RecordingTriggerEvent, handler=self._handle_recording_trigger)
        self._event_bus.subscribe(event_type=AudioModeChangeRequestEvent, handler=self._handle_audio_mode_change_request)
        
        logger.info("Audio service event subscriptions configured")

    async def _handle_recording_trigger(self, event: RecordingTriggerEvent):
        """Handle recording trigger event - recorders already running continuously"""
        if event.trigger == "start":
            logger.info("Start recording command received - recorders already active")
        elif event.trigger == "stop":
            logger.info("Stop recording command received - recorders continue running")
        else:
            logger.warning(f"Unknown recording trigger: {event.trigger}")



    async def _handle_audio_mode_change_request(self, event: AudioModeChangeRequestEvent):
        """Handle audio mode change requests - switches between command and dictation modes"""
        try:
            logger.info(f"Audio mode change request received: mode={event.mode}, reason={event.reason}")
            
            with self._lock:
                if event.mode == "dictation":
                    self._is_dictation_mode = True
                    # Both recorders active: command for amber detection, dictation for text
                    self._command_recorder.set_active(True)
                    self._dictation_recorder.set_active(True)
                    logger.info("Dictation mode: both recorders active")
                elif event.mode == "command":
                    self._is_dictation_mode = False
                    # Only command recorder active
                    self._command_recorder.set_active(True)
                    self._dictation_recorder.set_active(False)
                    logger.info("Command mode: only command recorder active")
                else:
                    logger.warning(f"Unknown audio mode requested: {event.mode}")

        except Exception as e:
            logger.error(f"Error handling audio mode change request: {e}", exc_info=True)



    def start_processing(self):
        """Start both recorders - they run continuously"""
        try:
            logger.info("Starting audio processing with dual recorders")
            
            # Start both recorders - they run continuously until shutdown
            self._command_recorder.start()
            self._dictation_recorder.start()
            
            # Set initial active states based on current mode
            with self._lock:
                if self._is_dictation_mode:
                    self._command_recorder.set_active(True)
                    self._dictation_recorder.set_active(True)
                    logger.info("Started in dictation mode: both recorders active")
                else:
                    self._command_recorder.set_active(True)
                    self._dictation_recorder.set_active(False)
                    logger.info("Started in command mode: only command recorder active")
            
            logger.info("Audio processing started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start audio processing: {e}", exc_info=True)
            raise

    def stop_processing(self):
        """Stop both recorders - only called during shutdown"""
        try:
            logger.info("Stopping audio processing")
            self._command_recorder.stop()
            self._dictation_recorder.stop()
            logger.info("Audio processing stopped")
        except Exception as e:
            logger.error(f"Error stopping audio processing: {e}", exc_info=True)


    async def shutdown(self):
        """Shutdown audio service"""
        try:
            logger.info("Shutting down audio service")
            self.stop_processing()
            logger.info("Audio service shutdown complete")
        except Exception as e:
            logger.error(f"Error during audio service shutdown: {e}", exc_info=True)

    def setup_subscriptions(self):
        self.init_listeners()
