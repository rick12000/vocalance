import asyncio
import logging
from typing import Callable, Optional

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.event_bus import EventBus
from vocalance.app.services.audio.audio_listeners import CommandAudioListener, SoundAudioListener
from vocalance.app.services.audio.audio_processor import AudioProcessor
from vocalance.app.services.audio.recorder import AudioRecorder
from vocalance.app.services.base_service import Service

logger = logging.getLogger(__name__)


class AudioService(Service):
    """Recorder → command/sound VAD listeners; dictation PCM via ``set_dictation_chunk_callback``."""

    def __init__(
        self, event_bus: EventBus, config: GlobalAppConfig, main_event_loop: Optional[asyncio.AbstractEventLoop] = None
    ) -> None:
        self._event_bus = event_bus
        self._config = config

        try:
            self._loop = main_event_loop or asyncio.get_running_loop()
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

        shared_processor = AudioProcessor(
            sample_rate=config.audio.sample_rate,
            enable_normalization=config.vad.enable_audio_normalization,
        )

        self._command_listener = CommandAudioListener(event_bus, config, shared_processor)
        self._sound_listener = SoundAudioListener(event_bus, config, shared_processor)
        self._sound_listener.setup_subscriptions()

        self._recorder = AudioRecorder(
            app_config=config,
            event_bus=event_bus,
            on_audio_chunk=self._on_audio_chunk,
            loop=self._loop,
        )

        self._dictation_chunk_callback: Optional[Callable[[bytes, int], None]] = None
        self._level_meter_callback: Optional[Callable[[bytes], None]] = None

    def set_dictation_chunk_callback(self, callback: Optional[Callable[[bytes, int], None]]) -> None:
        self._dictation_chunk_callback = callback

    def set_level_meter_callback(self, callback: Optional[Callable[[bytes], None]]) -> None:
        self._level_meter_callback = callback

    def _on_audio_chunk(self, audio_bytes: bytes, timestamp: float) -> None:
        if cb := self._dictation_chunk_callback:
            try:
                cb(audio_bytes, self._config.audio.sample_rate)
            except Exception as e:
                logger.error("Dictation chunk callback error: %s", e, exc_info=True)

        if lm := self._level_meter_callback:
            try:
                lm(audio_bytes)
            except Exception as e:
                logger.debug("Level meter callback error: %s", e)

        if self._command_listener:
            self._command_listener.process_audio_chunk(audio_bytes, timestamp)
        if self._sound_listener:
            self._sound_listener.process_audio_chunk(audio_bytes, timestamp)

    def start_processing(self) -> None:
        self._recorder.start()

    def stop_processing(self) -> None:
        self._recorder.stop()

    async def on_command_silent_chunks_updated(self, chunks: int) -> None:
        if self._command_listener:
            self._command_listener.update_silent_chunks_threshold(chunks)

    async def shutdown(self) -> None:
        try:
            self.stop_processing()
        except Exception as e:
            logger.error("Error stopping audio: %s", e, exc_info=True)
        self._dictation_chunk_callback = None
        self._level_meter_callback = None
        self._recorder = None
        self._command_listener = None
        self._sound_listener = None

    def get_recorder(self) -> AudioRecorder:
        return self._recorder
