import asyncio
import logging
import threading
import time
from typing import Optional

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.event_bus import EventBus
from vocalance.app.events.base_event import BaseEvent
from vocalance.app.events.core_events import (
    AudioDetectedEvent,
    CommandAudioSegmentReadyEvent,
    DictationAudioSegmentReadyEvent,
    RecordingTriggerEvent,
)
from vocalance.app.events.dictation_events import AudioModeChangeRequestEvent
from vocalance.app.services.audio.recorder import AudioRecorder

logger = logging.getLogger(__name__)


class AudioService:
    """Dual-recorder audio service with independent command and dictation streams.

    Manages two concurrent AudioRecorder instances optimized for different use cases:
    command recorder (low-latency, short utterances) and dictation recorder (longer
    duration, more tolerant of pauses). Dynamically activates/deactivates recorders
    based on mode to minimize resource usage. Publishes audio segment events to the
    event bus for downstream STT processing. Handles mode switching between command
    and dictation via event subscriptions.

    Attributes:
        _command_recorder: AudioRecorder optimized for command mode.
        _dictation_recorder: AudioRecorder optimized for dictation mode.
        _is_dictation_mode: Current mode flag (True=dictation, False=command).
        _main_event_loop: Asyncio event loop for publishing events from recorder threads.
    """

    def __init__(
        self, event_bus: EventBus, config: GlobalAppConfig, main_event_loop: Optional[asyncio.AbstractEventLoop] = None
    ) -> None:
        """Initialize audio service with dual recorders and event bus integration.

        Args:
            event_bus: EventBus for publishing audio segment events.
            config: Global application configuration with audio and VAD settings.
            main_event_loop: Optional asyncio event loop for thread-safe event publishing.
        """
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

        logger.debug("AudioService initialized with dual independent recorders")

    def _initialize_recorders(self) -> None:
        """Initialize both command and dictation audio recorders with callbacks.

        Creates two AudioRecorder instances with mode-specific configurations and
        registers callbacks for handling captured audio segments.

        Raises:
            Exception: If recorder initialization fails (propagated from AudioRecorder).
        """
        try:
            self._command_recorder = AudioRecorder(
                app_config=self._config,
                mode="command",
                on_audio_segment=self._on_command_audio_segment,
                on_audio_detected=self._on_audio_detected,
            )

            self._dictation_recorder = AudioRecorder(
                app_config=self._config, mode="dictation", on_audio_segment=self._on_dictation_audio_segment
            )

            logger.debug("Dual audio recorders initialized")

        except Exception as e:
            logger.error(f"Failed to initialize audio recorders: {e}", exc_info=True)
            raise

    def _on_command_audio_segment(self, segment_bytes: bytes) -> None:
        """Callback invoked when command recorder captures a complete audio segment.

        Args:
            segment_bytes: Raw audio bytes from the command recorder.
        """
        try:
            event = CommandAudioSegmentReadyEvent(audio_bytes=segment_bytes, sample_rate=self._config.audio.sample_rate)
            self._publish_audio_event(event)
            logger.debug(f"Command audio: {len(segment_bytes)} bytes")
        except Exception as e:
            logger.error(f"Error handling command audio: {e}")

    def _on_dictation_audio_segment(self, segment_bytes: bytes) -> None:
        """Callback invoked when dictation recorder captures a complete audio segment.

        Args:
            segment_bytes: Raw audio bytes from the dictation recorder.
        """
        try:
            logger.info(f"Publishing dictation audio segment: {len(segment_bytes)} bytes at {self._config.audio.sample_rate}Hz")
            event = DictationAudioSegmentReadyEvent(audio_bytes=segment_bytes, sample_rate=self._config.audio.sample_rate)
            self._publish_audio_event(event)
            logger.debug(f"Dictation audio: {len(segment_bytes)} bytes")
        except Exception as e:
            logger.error(f"Error handling dictation audio: {e}", exc_info=True)

    def _on_audio_detected(self) -> None:
        """Callback invoked immediately when command recorder detects speech onset.

        Used for triggering Markov prediction before STT completes.
        """
        try:
            event = AudioDetectedEvent(timestamp=time.time())
            self._publish_audio_event(event)
        except Exception as e:
            logger.error(f"Error handling audio detected: {e}")

    def _publish_audio_event(self, event_data: BaseEvent) -> None:
        """Publish event to event bus from recorder thread in a thread-safe manner.

        Uses run_coroutine_threadsafe to safely publish events from recorder threads
        to the main event loop. Handles closed event loops gracefully.

        Args:
            event_data: Event instance to publish.
        """
        if not self._main_event_loop or self._main_event_loop.is_closed():
            return

        try:
            asyncio.run_coroutine_threadsafe(self._event_bus.publish(event_data), self._main_event_loop)
        except RuntimeError as e:
            logger.debug(f"Event loop closed while publishing event: {e}")

    def init_listeners(self) -> None:
        """Register event subscriptions for audio service control.

        Subscribes to recording trigger and audio mode change events for controlling
        recorder behavior during runtime.
        """
        self._event_bus.subscribe(event_type=RecordingTriggerEvent, handler=self._handle_recording_trigger)
        self._event_bus.subscribe(event_type=AudioModeChangeRequestEvent, handler=self._handle_audio_mode_change_request)

        logger.info("Audio service event subscriptions configured")

    async def _handle_recording_trigger(self, event: RecordingTriggerEvent) -> None:
        """Handle recording trigger event (legacy - recorders run continuously).

        Args:
            event: Recording trigger event with start/stop command.
        """
        if event.trigger == "start":
            logger.info("Start recording command received - recorders already active")
        elif event.trigger == "stop":
            logger.info("Stop recording command received - recorders continue running")
        else:
            logger.warning(f"Unknown recording trigger: {event.trigger}")

    async def _handle_audio_mode_change_request(self, event: AudioModeChangeRequestEvent) -> None:
        """Handle audio mode change requests between command and dictation modes.

        Switches recorder activation states based on requested mode: command mode
        activates only command recorder, dictation mode activates both recorders
        to enable stop word detection during dictation.

        Args:
            event: Audio mode change request event with target mode and reason.
        """
        try:
            logger.info(f"Audio mode change request received: mode={event.mode}, reason={event.reason}")

            with self._lock:
                if event.mode == "dictation":
                    self._is_dictation_mode = True
                    if self._command_recorder:
                        self._command_recorder.set_active(True)
                    if self._dictation_recorder:
                        self._dictation_recorder.set_active(True)
                    logger.info("Dictation mode: both recorders active")
                elif event.mode == "command":
                    self._is_dictation_mode = False
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
        """Start both audio recorders and configure initial activation state.

        Starts both command and dictation recorder threads, then sets their activation
        states based on current mode (command mode = command only, dictation mode = both).
        """
        try:
            logger.info("Starting audio processing with dual recorders")

            if self._command_recorder:
                self._command_recorder.start()
            if self._dictation_recorder:
                self._dictation_recorder.start()

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
        """Stop both audio recorders and clean up their threads.

        Signals both recorders to stop, waits for thread termination, and releases
        audio stream resources. Thread-safe operation.
        """
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
        """Shutdown audio service with complete resource cleanup.

        Stops all recorders, waits for thread termination, and releases recorder
        references to enable garbage collection. Safe to call multiple times.
        """
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
        """Setup event subscriptions for the audio service.

        Delegates to init_listeners() for backward compatibility. Called by main
        initialization sequence after all services are created.
        """
        self.init_listeners()

    def on_dictation_silent_chunks_updated(self, chunks: int) -> None:
        """Update dictation silent chunks threshold dynamically during runtime.

        Allows real-time adjustment of silence detection sensitivity in dictation mode,
        forwarding the update to the dictation recorder instance.

        Args:
            chunks: New number of consecutive silent chunks required to end recording.
        """
        with self._lock:
            if self._dictation_recorder:
                self._dictation_recorder.update_dictation_silent_chunks(chunks)
            else:
                logger.warning("Dictation recorder not initialized, cannot update silent chunks")
