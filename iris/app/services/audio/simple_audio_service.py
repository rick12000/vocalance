import asyncio
import logging
import threading
import time
from typing import Optional

from iris.app.config.app_config import GlobalAppConfig
from iris.app.event_bus import EventBus
from iris.app.events.base_event import BaseEvent
from iris.app.events.core_events import (
    AudioDetectedEvent,
    CommandAudioSegmentReadyEvent,
    DictationAudioSegmentReadyEvent,
    RecordingTriggerEvent,
)
from iris.app.events.dictation_events import AudioModeChangeRequestEvent
from iris.app.services.audio.recorder import AudioRecorder

logger = logging.getLogger(__name__)


class SimpleAudioService:
    """Dual-recorder audio service with independent command and dictation streams.

    Manages separate optimized recorders for command (speed) and dictation (accuracy)
    modes with mode-based activation control for efficient resource usage.
    """

    def __init__(
        self, event_bus: EventBus, config: GlobalAppConfig, main_event_loop: Optional[asyncio.AbstractEventLoop] = None
    ) -> None:
        self._event_bus = event_bus
        self._config = config

        if main_event_loop:
            self._main_event_loop = main_event_loop
        else:
            try:
                self._main_event_loop = asyncio.get_running_loop()
            except RuntimeError:
                self._main_event_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._main_event_loop)

        self._is_dictation_mode: bool = False
        self._lock = threading.Lock()
        self._command_recorder: Optional[AudioRecorder] = None
        self._dictation_recorder: Optional[AudioRecorder] = None
        self._is_processing = False
        self._initialize_recorders()

        logger.info("SimpleAudioService initialized with dual independent recorders")

    def _initialize_recorders(self) -> None:
        try:
            # Command recorder - optimized for speed
            self._command_recorder = AudioRecorder(
                app_config=self._config,
                mode="command",
                on_audio_segment=self._on_command_audio_segment,
                on_audio_detected=self._on_audio_detected,
            )

            # Dictation recorder - optimized for accuracy
            self._dictation_recorder = AudioRecorder(
                app_config=self._config, mode="dictation", on_audio_segment=self._on_dictation_audio_segment
            )

            logger.info("Dual audio recorders initialized")

        except Exception as e:
            logger.error(f"Failed to initialize audio recorders: {e}", exc_info=True)
            raise

    def _on_command_audio_segment(self, segment_bytes: bytes) -> None:
        try:
            event = CommandAudioSegmentReadyEvent(audio_bytes=segment_bytes, sample_rate=self._config.audio.sample_rate)
            self._publish_audio_event(event)
            logger.debug(f"Command audio: {len(segment_bytes)} bytes")
        except Exception as e:
            logger.error(f"Error handling command audio: {e}")

    def _on_dictation_audio_segment(self, segment_bytes: bytes) -> None:
        try:
            logger.info(f"Publishing dictation audio segment: {len(segment_bytes)} bytes at {self._config.audio.sample_rate}Hz")
            event = DictationAudioSegmentReadyEvent(audio_bytes=segment_bytes, sample_rate=self._config.audio.sample_rate)
            self._publish_audio_event(event)
            logger.debug(f"Dictation audio: {len(segment_bytes)} bytes")
        except Exception as e:
            logger.error(f"Error handling dictation audio: {e}", exc_info=True)

    def _on_audio_detected(self) -> None:
        try:
            event = AudioDetectedEvent(timestamp=time.time())
            self._publish_audio_event(event)
        except Exception as e:
            logger.error(f"Error handling audio detected: {e}")

    def _publish_audio_event(self, event_data: BaseEvent) -> None:
        if not self._main_event_loop or self._main_event_loop.is_closed():
            return

        try:
            asyncio.run_coroutine_threadsafe(self._event_bus.publish(event_data), self._main_event_loop)
        except RuntimeError as e:
            logger.debug(f"Event loop closed while publishing event: {e}")

    def init_listeners(self) -> None:
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
                    if self._command_recorder:
                        self._command_recorder.set_active(True)
                    if self._dictation_recorder:
                        self._dictation_recorder.set_active(True)
                    logger.info("Dictation mode: both recorders active")
                elif event.mode == "command":
                    self._is_dictation_mode = False
                    # Only command recorder active
                    if self._command_recorder:
                        self._command_recorder.set_active(True)
                    if self._dictation_recorder:
                        self._dictation_recorder.set_active(False)
                    logger.info("Command mode: only command recorder active")
                else:
                    logger.warning(f"Unknown audio mode requested: {event.mode}")

        except Exception as e:
            logger.error(f"Error handling audio mode change request: {e}", exc_info=True)

    def start_processing(self) -> None:
        try:
            logger.info("Starting audio processing with dual recorders")

            # Start both recorders - they run continuously until shutdown
            if self._command_recorder:
                self._command_recorder.start()
            if self._dictation_recorder:
                self._dictation_recorder.start()

            # Set initial active states based on current mode
            with self._lock:
                if self._is_dictation_mode:
                    if self._command_recorder:
                        self._command_recorder.set_active(True)
                    if self._dictation_recorder:
                        self._dictation_recorder.set_active(True)
                    logger.info("Started in dictation mode: both recorders active")
                else:
                    if self._command_recorder:
                        self._command_recorder.set_active(True)
                    if self._dictation_recorder:
                        self._dictation_recorder.set_active(False)
                    logger.info("Started in command mode: only command recorder active")

            logger.info("Audio processing started successfully")

        except Exception as e:
            logger.error(f"Failed to start audio processing: {e}", exc_info=True)
            raise

    def stop_processing(self) -> None:
        """Stop audio processing - properly cleans up recorder threads"""
        try:
            logger.info("Stopping audio processing")

            with self._lock:
                if self._command_recorder:
                    self._command_recorder.stop()
                if self._dictation_recorder:
                    self._dictation_recorder.stop()

            logger.info("Audio processing stopped")
        except Exception as e:
            logger.error(f"Error stopping audio processing: {e}", exc_info=True)

    async def shutdown(self) -> None:
        """Shutdown audio service with proper resource cleanup"""
        try:
            logger.info("Shutting down audio service")
            self.stop_processing()

            # Explicitly release recorder references
            with self._lock:
                self._command_recorder = None
                self._dictation_recorder = None

            logger.info("Audio service shutdown complete")
        except Exception as e:
            logger.error(f"Error during audio service shutdown: {e}", exc_info=True)

    def setup_subscriptions(self) -> None:
        self.init_listeners()
