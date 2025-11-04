import asyncio
import logging
from typing import Optional

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.event_bus import EventBus
from vocalance.app.events.core_events import AudioChunkEvent, RecordingTriggerEvent
from vocalance.app.events.dictation_events import AudioModeChangeRequestEvent
from vocalance.app.services.audio.audio_listeners import CommandAudioListener, DictationAudioListener, SoundAudioListener
from vocalance.app.services.audio.recorder import AudioRecorder

logger = logging.getLogger(__name__)


class AudioService:
    """Unified audio service with continuous streaming architecture.

    Manages a single AudioRecorder that continuously streams 50ms audio chunks.
    Three independent listeners (CommandAudioListener, DictationAudioListener,
    SoundAudioListener) subscribe to the chunk stream and apply their own logic:

    - Command: ~150ms silence timeout for low-latency stop word detection
    - Dictation: ~800ms silence timeout for full content capture
    - Sound: Accumulates 100ms buffers (rate-limited to 10/sec) for sound recognition

    This architecture eliminates resource duplication while enabling simultaneous
    segment detection with different parameters. All listeners process the same
    audio stream in parallel, each emitting their respective events independently.

    Attributes:
        _recorder: AudioRecorder for continuous chunk streaming.
        _command_listener: CommandAudioListener for command segment detection.
        _dictation_listener: DictationAudioListener for dictation segment detection.
        _sound_listener: SoundAudioListener for sound recognition with rate limiting.
        _main_event_loop: Asyncio event loop for publishing events from recorder thread.
    """

    def __init__(
        self, event_bus: EventBus, config: GlobalAppConfig, main_event_loop: Optional[asyncio.AbstractEventLoop] = None
    ) -> None:
        """Initialize audio service with recorder and listeners.

        Args:
            event_bus: EventBus for publishing AudioChunkEvent and managing subscriptions.
            config: Global application configuration.
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

        # Create continuous audio recorder
        self._recorder = AudioRecorder(
            app_config=config,
            on_audio_chunk=self._on_audio_chunk_callback,
        )

        # Create audio listeners
        self._command_listener = CommandAudioListener(event_bus, config)
        self._dictation_listener = DictationAudioListener(event_bus, config)
        self._sound_listener = SoundAudioListener(event_bus, config)

        logger.debug("AudioService initialized with continuous streaming architecture (3 listeners)")

    def _on_audio_chunk_callback(self, audio_bytes: bytes, timestamp: float) -> None:
        """Callback from recorder for each 50ms audio chunk.

        Publishes AudioChunkEvent to event bus for downstream listeners to process.
        This bridges the synchronous recorder thread to the async event bus.

        Uses run_coroutine_threadsafe for proper exception handling and task tracking.

        Args:
            audio_bytes: Raw audio data for this chunk (int16 format).
            timestamp: Timestamp when chunk was captured.
        """
        try:
            event = AudioChunkEvent(
                audio_chunk=audio_bytes,
                sample_rate=self._config.audio.sample_rate,
                timestamp=timestamp,
            )
            # Use run_coroutine_threadsafe for proper exception propagation
            future = asyncio.run_coroutine_threadsafe(self._event_bus.publish(event), self._main_event_loop)
            # Add exception callback to log any errors
            future.add_done_callback(self._handle_publish_result)
        except RuntimeError as e:
            logger.debug(f"Event loop closed while publishing audio chunk: {e}")
        except Exception as e:
            logger.error(f"Error publishing audio chunk: {e}", exc_info=True)

    def _handle_publish_result(self, future: asyncio.Future) -> None:
        """Handle result of audio chunk publish operation.

        Logs any exceptions that occurred during publishing.

        Args:
            future: Completed future from run_coroutine_threadsafe.
        """
        try:
            # Check if exception occurred
            future.result()
        except Exception as e:
            logger.error(f"Error in audio chunk publish: {e}", exc_info=True)

    def init_listeners(self) -> None:
        """Register event subscriptions for audio service control.

        Sets up:
        - CommandAudioListener subscription to AudioChunkEvent
        - DictationAudioListener subscription to AudioChunkEvent
        - SoundAudioListener subscription to AudioChunkEvent + DictationModeDisableOthersEvent
        - AudioService subscriptions to control events
        """
        # Setup listener subscriptions
        self._command_listener.setup_subscriptions()
        self._dictation_listener.setup_subscriptions()
        self._sound_listener.setup_subscriptions()

        # Setup service control subscriptions
        self._event_bus.subscribe(event_type=RecordingTriggerEvent, handler=self._handle_recording_trigger)
        self._event_bus.subscribe(event_type=AudioModeChangeRequestEvent, handler=self._handle_audio_mode_change_request)

        logger.info("Audio service event subscriptions configured (3 listeners)")

    async def _handle_recording_trigger(self, event: RecordingTriggerEvent) -> None:
        """Handle recording trigger event (legacy - recorder runs continuously).

        Args:
            event: Recording trigger event with start/stop command.
        """
        if event.trigger == "start":
            logger.info("Start recording command received - recorder already active")
        elif event.trigger == "stop":
            logger.info("Stop recording command received - recorder continues running")
        else:
            logger.warning(f"Unknown recording trigger: {event.trigger}")

    async def _handle_audio_mode_change_request(self, event: AudioModeChangeRequestEvent) -> None:
        """Handle audio mode change requests between command and dictation modes.

        Mode switching is passive in this architecture - both listeners are always
        active and process chunks independently. Mode changes are primarily for
        downstream services (STT, etc).

        Args:
            event: Audio mode change request event with target mode and reason.
        """
        try:
            logger.info(f"Audio mode change request received: mode={event.mode}, reason={event.reason}")
            # In continuous streaming architecture, both listeners are always active
            # Mode switching is handled by downstream services (STT, dictation coordinator)
            logger.debug("Mode change acknowledged (both listeners remain active)")

        except Exception as e:
            logger.error(f"Error handling audio mode change request: {e}", exc_info=True)

    def start_processing(self) -> None:
        """Start the audio recorder for continuous chunk streaming.

        Starts the recorder thread which will continuously capture and publish
        50ms audio chunks to the event bus. Listeners are already subscribed
        and will begin processing chunks immediately.
        """
        try:
            logger.info("Starting audio processing with continuous streaming")
            self._recorder.start()
            logger.info("Audio processing started successfully")

        except Exception as e:
            logger.error(f"Failed to start audio processing: {e}", exc_info=True)
            raise

    def stop_processing(self) -> None:
        """Stop the audio recorder and clean up resources.

        Signals the recorder to stop, waits for thread termination, and releases
        audio stream resources. Thread-safe operation.
        """
        try:
            logger.info("Stopping audio processing")
            self._recorder.stop()
            logger.info("Audio processing stopped")
        except Exception as e:
            logger.error(f"Error stopping audio processing: {e}", exc_info=True)

    async def shutdown(self) -> None:
        """Shutdown audio service with complete resource cleanup.

        Stops recorder, waits for thread termination, and releases references
        to enable garbage collection. Safe to call multiple times.
        """
        try:
            logger.info("Shutting down audio service")
            self.stop_processing()

            # Release references
            self._recorder = None
            self._command_listener = None
            self._dictation_listener = None
            self._sound_listener = None

            logger.info("Audio service shutdown complete")
        except Exception as e:
            logger.error(f"Error during audio service shutdown: {e}", exc_info=True)

    def setup_subscriptions(self) -> None:
        """Setup event subscriptions for the audio service.

        Delegates to init_listeners() for backward compatibility. Called by main
        initialization sequence after all services are created.
        """
        self.init_listeners()

    async def on_dictation_silent_chunks_updated(self, chunks: int) -> None:
        """Update dictation silent chunks threshold dynamically during runtime.

        Allows real-time adjustment of silence detection sensitivity in dictation mode,
        forwarding the update to the dictation listener instance.

        Thread-safe: Delegates to listener's async method with lock protection.

        Args:
            chunks: New number of consecutive silent chunks required to end recording.
        """
        if self._dictation_listener:
            await self._dictation_listener.update_silent_chunks_threshold(chunks)
            logger.info(f"Updated dictation silent chunks to {chunks}")
        else:
            logger.warning("Dictation listener not initialized, cannot update silent chunks")
