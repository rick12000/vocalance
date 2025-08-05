"""
Streamlined Audio Service

Manages dual independent audio recorders for command and dictation modes.
Provides clear separation of concerns and optimized threading.
"""

import logging
import threading
import asyncio
from typing import Optional

from iris.event_bus import EventBus
from iris.services.audio.recorder import AudioRecorder

from iris.events.core_events import (
    CommandAudioSegmentReadyEvent,
    DictationAudioSegmentReadyEvent,
    StartRecordingCommand,
    StopRecordingCommand,
    AudioRecordingStateEvent,
    RequestAudioSampleForTrainingCommand,
    AudioSampleForTrainingReadyEvent
)
from iris.events.base_event import BaseEvent
from iris.events.dictation_events import DictationModeDisableOthersEvent, AudioModeChangeRequestEvent
from iris.config.app_config import GlobalAppConfig


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
            # Command recorder - optimized for speed with streaming support
            self._command_recorder = AudioRecorder(
                app_config=self._config,
                mode="command",
                on_audio_segment=self._on_command_audio_segment,
                on_streaming_chunk=self._on_command_streaming_chunk  # New: streaming callback
            )
            
            # Dictation recorder - optimized for accuracy
            self._dictation_recorder = AudioRecorder(
                app_config=self._config,
                mode="dictation", 
                on_audio_segment=self._on_dictation_audio_segment
            )
            
            logger.info("Dual audio recorders initialized with streaming optimization")
            
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

    def _on_command_streaming_chunk(self, audio_bytes: bytes, is_final: bool) -> str:
        """Handle streaming command recognition for early termination"""
        try:
            # Direct STT processing for ultra-low latency - bypass event bus
            # This is a fast path for instant command recognition
            if hasattr(self, '_streaming_stt_engine'):
                recognized_text = self._streaming_stt_engine.recognize_streaming(audio_bytes, is_final)
                if recognized_text:
                    logger.debug(f"Streaming recognition: '{recognized_text}'")
                    return recognized_text
            return ""
        except Exception as e:
            logger.error(f"Error in streaming chunk processing: {e}")
            return ""
    
    def _publish_audio_event(self, event_data: BaseEvent):
        """Unified event publication method"""
        if self._main_event_loop and not self._main_event_loop.is_closed():
            asyncio.run_coroutine_threadsafe(
                self._event_bus.publish(event_data),
                self._main_event_loop
            )

    def init_listeners(self):
        """Subscribe to relevant events."""
        self._event_bus.subscribe(StartRecordingCommand, self._handle_start_recording)
        self._event_bus.subscribe(StopRecordingCommand, self._handle_stop_recording)
        self._event_bus.subscribe(AudioModeChangeRequestEvent, self._handle_audio_mode_change_request)
        
        logger.info("Audio service event subscriptions configured")

    async def _handle_start_recording(self, event: StartRecordingCommand):
        """Handle start recording command - recorders already running continuously"""
        logger.info("Start recording command received - recorders already active")

    async def _handle_stop_recording(self, event: StopRecordingCommand):
        """Handle stop recording command - recorders keep running, only stop during shutdown"""
        logger.info("Stop recording command received - recorders continue running")



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

            # Legacy sound manager callback method removed - no longer needed with StreamlinedSoundService

    def get_status(self) -> dict:
        """Get current audio service status"""
        return {
            "command_recorder_active": self._command_recorder.is_active() if self._command_recorder else False,
            "dictation_recorder_active": self._dictation_recorder.is_active() if self._dictation_recorder else False,
            "command_recording": self._command_recorder.is_recording() if self._command_recorder else False,
            "dictation_recording": self._dictation_recorder.is_recording() if self._dictation_recorder else False,
            "dictation_mode": self._is_dictation_mode
        }

    async def shutdown(self):
        """Shutdown audio service"""
        try:
            logger.info("Shutting down audio service")
            self.stop_processing()
            logger.info("Audio service shutdown complete")
        except Exception as e:
            logger.error(f"Error during audio service shutdown: {e}", exc_info=True)

    def set_streaming_stt_engine(self, stt_engine):
        """Set the STT engine for streaming recognition (called by STT service)"""
        self._streaming_stt_engine = stt_engine
        logger.info("Streaming STT engine configured for direct processing")

    def setup_subscriptions(self):
        self.init_listeners()
