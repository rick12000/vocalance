import asyncio
import logging
from typing import Callable, Optional

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.event_bus import EventBus
from vocalance.app.events.dictation_events import AudioModeChangeRequestEvent
from vocalance.app.services.audio.audio_listeners import CommandAudioListener, SoundAudioListener
from vocalance.app.services.audio.audio_processor import AudioProcessor
from vocalance.app.services.audio.recorder import AudioRecorder

logger = logging.getLogger(__name__)


class AudioService:
    """Recorder → command/sound listeners; dictation PCM via ``set_dictation_chunk_callback``."""

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

        self._recorder = AudioRecorder(
            app_config=config,
            event_bus=event_bus,
            on_audio_chunk=self._on_audio_chunk_callback,
            loop=self._main_event_loop,
        )

        self._shared_audio_processor = AudioProcessor(
            sample_rate=config.audio.sample_rate,
            enable_normalization=config.vad.enable_audio_normalization,
        )

        self._command_listener = CommandAudioListener(event_bus, config, self._shared_audio_processor)
        self._sound_listener = SoundAudioListener(event_bus, config, self._shared_audio_processor)

        self._dictation_chunk_callback: Optional[Callable[[bytes, int], None]] = None

        self._level_meter_callback: Optional[Callable[[bytes], None]] = None

    def set_dictation_chunk_callback(self, callback: Optional[Callable[[bytes, int], None]]) -> None:
        """Raw PCM hook from the recorder thread (dictation / Moonshine); keep it non-blocking."""
        self._dictation_chunk_callback = callback

    def set_level_meter_callback(self, callback: Optional[Callable[[bytes], None]]) -> None:
        """Optional UI level meter hook (raw bytes per chunk)."""
        self._level_meter_callback = callback

    def _on_audio_chunk_callback(self, audio_bytes: bytes, timestamp: float) -> None:
        sample_rate = self._config.audio.sample_rate
        cb = self._dictation_chunk_callback
        if cb is not None:
            try:
                cb(audio_bytes, sample_rate)
            except Exception as e:
                logger.error("dictation chunk callback error: %s", e, exc_info=True)

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
            logger.error("Audio listener chunk error: %s", e, exc_info=True)

    def init_listeners(self) -> None:
        self._command_listener.setup_subscriptions()
        self._sound_listener.setup_subscriptions()
        self._command_listener.set_main_event_loop(self._main_event_loop)
        self._sound_listener.set_main_event_loop(self._main_event_loop)

        self._event_bus.subscribe(event_type=AudioModeChangeRequestEvent, handler=self._handle_audio_mode_change_request)

        logger.info("Audio service event subscriptions configured (2 listeners)")

    def _handle_audio_mode_change_request(self, mode_change_request: AudioModeChangeRequestEvent) -> None:
        logger.info("Audio mode change request: mode=%s reason=%s", mode_change_request.mode, mode_change_request.reason)

    def start_processing(self) -> None:
        try:
            logger.info("Starting audio processing with continuous streaming")
            self._recorder.start()
            logger.info("Audio processing started successfully")

        except Exception as e:
            logger.error(f"Failed to start audio processing: {e}", exc_info=True)
            raise

    def stop_processing(self) -> None:
        try:
            logger.info("Stopping audio processing")
            self._recorder.stop()
            logger.info("Audio processing stopped")
        except Exception as e:
            logger.error(f"Error stopping audio processing: {e}", exc_info=True)

    def shutdown(self) -> None:
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
        self.init_listeners()

    async def on_command_silent_chunks_updated(self, chunks: int) -> None:
        if self._command_listener:
            await self._command_listener.update_silent_chunks_threshold(chunks)
            logger.info(f"Updated command silent chunks to {chunks}")
        else:
            logger.warning("Command listener not initialized, cannot update silent chunks")

    def get_recorder(self) -> AudioRecorder:
        return self._recorder
