import asyncio
import logging
import queue
import threading
from typing import Callable, Optional

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.event_bus import EventBus
from vocalance.app.events.core_events import RecordingTriggerEvent
from vocalance.app.events.dictation_events import AudioModeChangeRequestEvent
from vocalance.app.services.audio.audio_listeners import CommandAudioListener, SoundAudioListener
from vocalance.app.services.audio.audio_processor import AudioProcessor
from vocalance.app.services.audio.recorder import AudioRecorder

logger = logging.getLogger(__name__)


class AudioService:
    """Continuous capture: recorder → VAD worker thread → command/sound listeners.

    Per-chunk PCM stays off the asyncio event bus. Dictation uses
    ``set_dictation_chunk_callback`` (Moonshine ingress); simple-listening UI uses
    ``set_level_meter_callback``.
    """

    def __init__(
        self, event_bus: EventBus, config: GlobalAppConfig, main_event_loop: Optional[asyncio.AbstractEventLoop] = None
    ) -> None:
        """Wire recorder, shared ``AudioProcessor``, VAD queue/worker, and command/sound listeners."""
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

        self._recorder = AudioRecorder(
            app_config=config,
            event_bus=event_bus,
            on_audio_chunk=self._on_audio_chunk_callback,
        )

        self._shared_audio_processor = AudioProcessor(
            sample_rate=config.audio.sample_rate,
            enable_normalization=config.vad.enable_audio_normalization,
        )

        self._command_listener = CommandAudioListener(event_bus, config, self._shared_audio_processor)
        self._sound_listener = SoundAudioListener(event_bus, config, self._shared_audio_processor)

        self._dictation_chunk_callback: Optional[Callable[[bytes, int], None]] = None

        self._vad_queue: queue.Queue[tuple[bytes, float, int]] = queue.Queue()
        self._vad_stop = threading.Event()
        self._vad_worker_thread: Optional[threading.Thread] = None
        self._level_meter_callback: Optional[Callable[[bytes], None]] = None

        logger.debug("AudioService initialized with shared AudioProcessor and 2 listeners")

    def set_dictation_chunk_callback(self, callback: Optional[Callable[[bytes, int], None]]) -> None:
        """Register a non-blocking hook for raw PCM while dictation is active (Moonshine ingress).

        Invoked synchronously from the recorder thread for every chunk. Must not block the recorder
        for long; the coordinator queues PCM on a dedicated ingress thread.
        """
        self._dictation_chunk_callback = callback

    def set_level_meter_callback(self, callback: Optional[Callable[[bytes], None]]) -> None:
        """Optional hook for UI level meter (e.g. simple dictation listening)."""
        self._level_meter_callback = callback

    def _on_audio_chunk_callback(self, audio_bytes: bytes, timestamp: float) -> None:
        """Callback from recorder for each audio chunk."""
        sample_rate = self._config.audio.sample_rate
        cb = self._dictation_chunk_callback
        if cb is not None:
            try:
                cb(audio_bytes, sample_rate)
            except Exception as e:
                logger.error("dictation chunk callback error: %s", e, exc_info=True)
        try:
            self._vad_queue.put((audio_bytes, timestamp, sample_rate))
        except Exception as e:
            logger.error("VAD queue put failed: %s", e, exc_info=True)

    def _vad_worker_loop(self) -> None:
        """Drain VAD queue: optional level meter, then command/sound ``process_audio_chunk``."""
        while True:
            try:
                item = self._vad_queue.get(timeout=0.25)
            except queue.Empty:
                if self._vad_stop.is_set():
                    break
                continue
            audio_bytes, timestamp, _ = item
            lm = self._level_meter_callback
            if lm is not None:
                try:
                    lm(audio_bytes)
                except Exception as e:
                    logger.debug("level meter callback error: %s", e)
            try:
                if self._command_listener is not None:
                    self._command_listener.process_audio_chunk(audio_bytes, timestamp)
                if self._sound_listener is not None:
                    self._sound_listener.process_audio_chunk(audio_bytes, timestamp)
            except Exception as e:
                logger.error("VAD worker chunk error: %s", e, exc_info=True)
        logger.debug("Audio VAD worker thread exiting")

    def _start_vad_worker(self) -> None:
        """Start the daemon VAD worker thread if not already running."""
        if self._vad_worker_thread is not None and self._vad_worker_thread.is_alive():
            return
        self._vad_stop.clear()
        self._vad_worker_thread = threading.Thread(target=self._vad_worker_loop, name="AudioVADWorker", daemon=True)
        self._vad_worker_thread.start()
        logger.debug("Audio VAD worker thread started")

    def _stop_vad_worker(self) -> None:
        """Signal the VAD worker to stop, join it, and drain the queue."""
        self._vad_stop.set()
        if self._vad_worker_thread is not None:
            self._vad_worker_thread.join(timeout=5.0)
            if self._vad_worker_thread.is_alive():
                logger.warning("Audio VAD worker did not stop within timeout")
            self._vad_worker_thread = None
        while True:
            try:
                self._vad_queue.get_nowait()
            except queue.Empty:
                break

    def init_listeners(self) -> None:
        """Subscribe listeners, set the main loop for thread-safe publishes, and wire control events."""
        self._command_listener.setup_subscriptions()
        self._sound_listener.setup_subscriptions()
        self._command_listener.set_main_event_loop(self._main_event_loop)
        self._sound_listener.set_main_event_loop(self._main_event_loop)

        self._event_bus.subscribe(event_type=RecordingTriggerEvent, handler=self._handle_recording_trigger)
        self._event_bus.subscribe(event_type=AudioModeChangeRequestEvent, handler=self._handle_audio_mode_change_request)

        logger.info("Audio service event subscriptions configured (2 listeners)")

    async def _handle_recording_trigger(self, event: RecordingTriggerEvent) -> None:
        """Handle recording trigger event (recorder runs continuously; triggers are informational).

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

        Mode switching is passive: command and sound listeners stay subscribed; dictation
        uses the raw chunk stream via DictationCoordinator.

        Args:
            event: Audio mode change request event with target mode and reason.
        """
        try:
            logger.info(f"Audio mode change request received: mode={event.mode}, reason={event.reason}")
            logger.debug("Mode change acknowledged (listeners unchanged; downstream handles mode)")

        except Exception as e:
            logger.error(f"Error handling audio mode change request: {e}", exc_info=True)

    def start_processing(self) -> None:
        """Start the VAD worker and continuous recorder (~30 ms PCM chunks enqueued for VAD)."""
        try:
            logger.info("Starting audio processing with continuous streaming")
            self._start_vad_worker()
            self._recorder.start()
            logger.info("Audio processing started successfully")

        except Exception as e:
            logger.error(f"Failed to start audio processing: {e}", exc_info=True)
            raise

    def stop_processing(self) -> None:
        """Stop the recorder and the VAD worker thread."""
        try:
            logger.info("Stopping audio processing")
            self._recorder.stop()
            self._stop_vad_worker()
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
            self._level_meter_callback = None

            self._recorder = None
            self._command_listener = None
            self._sound_listener = None

            logger.info("Audio service shutdown complete")
        except Exception as e:
            logger.error(f"Error during audio service shutdown: {e}", exc_info=True)

    def setup_subscriptions(self) -> None:
        """Alias for ``init_listeners`` (startup wiring)."""
        self.init_listeners()

    async def on_command_silent_chunks_updated(self, chunks: int) -> None:
        """Update command silent chunks threshold dynamically during runtime.

        Allows real-time adjustment of silence detection sensitivity in command mode,
        forwarding the update to the command listener instance.

        Thread-safe: Delegates to listener's async method with lock protection.

        Args:
            chunks: New number of consecutive silent chunks required to end recording.
        """
        if self._command_listener:
            await self._command_listener.update_silent_chunks_threshold(chunks)
            logger.info(f"Updated command silent chunks to {chunks}")
        else:
            logger.warning("Command listener not initialized, cannot update silent chunks")

    def get_recorder(self) -> AudioRecorder:
        """Get the underlying audio recorder instance.

        Returns:
            AudioRecorder instance used by this service.
        """
        return self._recorder
