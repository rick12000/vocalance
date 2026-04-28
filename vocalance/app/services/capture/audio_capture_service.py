from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Optional

import numpy as np
import sounddevice as sd
from numpy.typing import NDArray

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.event_bus import EventBus
from vocalance.app.events.core_events import AudioChunkCapturedEvent, AudioDeviceErrorEvent
from vocalance.app.services.base_service import Service

logger = logging.getLogger(__name__)


class AudioCaptureService(Service):
    """Source of microphone audio for the rest of the application.

    Owns a single PortAudio input stream. For every PCM buffer the device
    delivers, publishes one :class:`AudioChunkCapturedEvent` on the bus.
    Surfaces a single :class:`AudioDeviceErrorEvent` if the device cannot be
    opened. Has no dependencies on, and no knowledge of, any consumer:
    segmenters, the dictation coordinator, and the UI wave-meter all receive
    chunks through ordinary bus subscriptions.
    """

    def __init__(
        self,
        event_bus: EventBus,
        config: GlobalAppConfig,
        main_event_loop: asyncio.AbstractEventLoop,
    ) -> None:
        super().__init__(event_bus)
        self.config = config
        self.loop = main_event_loop

        self.sample_rate: int = int(config.audio.sample_rate)
        chunk_seconds = float(config.audio.capture_chunk_duration_seconds)
        self.chunk_size: int = int(self.sample_rate * chunk_seconds)

        self._stream: Optional[sd.InputStream] = None
        self._recording = False
        self._state_lock = threading.Lock()

        self._device_error_published = False
        self._launch_input_device_name: Optional[str] = None

    def start(self) -> None:
        """Open the input stream and begin publishing chunks on the bus."""
        with self._state_lock:
            if self._recording:
                return
            self._recording = True

        if not self._open_stream():
            logger.error("Audio capture could not be started")
            with self._state_lock:
                self._recording = False
            self._publish_device_error(
                self.config.audio.device_capture_messages.message_for_launch_device(self._launch_input_device_name)
            )

    def stop(self) -> None:
        """Stop the input stream. Safe to call multiple times."""
        with self._state_lock:
            was_recording = self._recording
            self._recording = False

        if was_recording:
            self._close_stream()

    async def shutdown(self) -> None:
        self.stop()
        await super().shutdown()

    def _open_stream(self) -> bool:
        try:
            stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                channels=1,
                dtype="int16",
                device=None,
                callback=self._portaudio_callback,
            )
            stream.start()
            self._stream = stream
            self._capture_input_device_name()
            return True
        except Exception as e:
            logger.error("Failed to open audio input stream: %s", e)
            self._close_stream()
            return False

    def _capture_input_device_name(self) -> None:
        if self._launch_input_device_name is not None:
            return
        try:
            info = sd.query_devices(kind="input")
            name = info.get("name") if isinstance(info, dict) else None
            if name:
                self._launch_input_device_name = str(name)
        except Exception as e:
            logger.debug("Could not query default input device name: %s", e)

    def _close_stream(self) -> None:
        stream = self._stream
        if stream is None:
            return
        try:
            if stream.active:
                stream.stop()
            stream.close()
        except Exception as e:
            logger.warning("Error while closing audio stream: %s", e)
        finally:
            self._stream = None

    def _portaudio_callback(
        self,
        indata: NDArray[np.int16],
        frames: int,
        time_info: sd.CallbackTimeInfo,
        status: sd.CallbackFlags,
    ) -> None:
        """PortAudio thread entry point: copy bytes and hop to the main loop."""
        if status:
            logger.debug("Input stream status: %s", status)

        with self._state_lock:
            if not self._recording:
                return

        pcm_bytes = indata.tobytes()
        timestamp = time.time()

        try:
            self.loop.call_soon_threadsafe(self._publish_chunk, pcm_bytes, timestamp)
        except RuntimeError as e:
            logger.error("Could not schedule audio chunk: %s", e)

    def _publish_chunk(self, pcm_bytes: bytes, timestamp: float) -> None:
        """Main-loop entry point: publish a captured chunk on the bus."""
        asyncio.create_task(
            self.event_bus.publish(AudioChunkCapturedEvent(pcm_bytes=pcm_bytes, timestamp=timestamp, sample_rate=self.sample_rate))
        )

    def _publish_device_error(self, message: str) -> None:
        if self._device_error_published:
            return
        self._device_error_published = True
        asyncio.create_task(self.event_bus.publish(AudioDeviceErrorEvent(error_message=message)))
