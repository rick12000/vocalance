import asyncio
import logging
import threading
import time
from typing import Any, Callable, Optional

import numpy as np
import sounddevice as sd
from numpy.typing import NDArray

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.event_bus import EventBus
from vocalance.app.events.core_events import AudioDeviceErrorEvent


class AudioRecorder:
    """Single PortAudio input stream; forwards PCM chunks to ``on_audio_chunk`` on the asyncio loop."""

    def __init__(
        self,
        app_config: GlobalAppConfig,
        loop: asyncio.AbstractEventLoop,
        event_bus: EventBus,
        on_audio_chunk: Callable[[bytes, float], None],
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.loop = loop
        self.event_bus = event_bus
        self.on_audio_chunk = on_audio_chunk
        self.app_config = app_config

        self.sample_rate: int = int(app_config.audio.sample_rate)
        chunk_seconds = float(app_config.audio.capture_chunk_duration_seconds)
        self.chunk_size: int = int(self.sample_rate * chunk_seconds)

        self.device_error_already_published = False
        self.launch_input_device_name: Optional[str] = None

        self.recording = False
        self.stream: Optional[sd.InputStream] = None
        self.capture_state_lock = threading.Lock()
        self._inflight_deliveries = 0
        self._inflight_lock = threading.Lock()

    def portaudio_callback(
        self,
        indata: NDArray[np.int16],
        frames: int,
        time_info: Any,
        status: Optional[Any],
    ) -> None:
        if status:
            self.logger.debug("Input stream status: %s", status)

        with self.capture_state_lock:
            active = self.recording

        if active:
            audio_bytes = indata.tobytes()
            timestamp = time.time()

            def deliver() -> None:
                try:
                    self.on_audio_chunk(audio_bytes, timestamp)
                finally:
                    with self._inflight_lock:
                        self._inflight_deliveries -= 1

            with self._inflight_lock:
                self._inflight_deliveries += 1
            try:
                self.loop.call_soon_threadsafe(deliver)
            except RuntimeError as e:
                self.logger.error("Could not schedule audio chunk: %s", e)
                with self._inflight_lock:
                    self._inflight_deliveries -= 1

    def schedule_device_error_publish_on_loop(self, message: str) -> None:
        async def publish() -> None:
            await self.event_bus.publish(AudioDeviceErrorEvent(error_message=message))

        try:
            self.loop.create_task(publish())
        except RuntimeError as e:
            self.logger.error("Could not schedule device error publish: %s", e)

    def open_input_stream(self) -> bool:
        try:
            stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                channels=1,
                dtype="int16",
                device=None,
                callback=self.portaudio_callback,
            )
            stream.start()
            self.stream = stream
            if self.launch_input_device_name is None:
                try:
                    info = sd.query_devices(kind="input")
                    name = info.get("name") if isinstance(info, dict) else None
                    if name:
                        self.launch_input_device_name = str(name)
                except Exception as e:
                    self.logger.debug("Could not query default input device name: %s", e)
            return True
        except Exception as e:
            self.logger.error("Failed to open audio input stream: %s", e)
            self.close_stream_resources()
            return False

    def publish_device_error_once(self, error_message: str) -> None:
        if not self.device_error_already_published:
            self.device_error_already_published = True
            try:
                self.loop.call_soon_threadsafe(self.schedule_device_error_publish_on_loop, error_message)
            except RuntimeError as e:
                self.logger.error("Could not schedule device error: %s", e)

    def close_stream_resources(self) -> None:
        stream = self.stream
        if stream is not None:
            try:
                if stream.active:
                    stream.stop()
                stream.close()
            except Exception as e:
                self.logger.warning("Error while closing audio stream: %s", e)
            finally:
                self.stream = None

    def start(self) -> None:
        with self.capture_state_lock:
            already_recording = self.recording
            if not already_recording:
                self.recording = True

        if not already_recording:
            stream_ok = self.open_input_stream()
            if not stream_ok:
                self.logger.error("Audio capture could not be started")
                msg = self.app_config.audio.device_capture_messages.message_for_launch_device(self.launch_input_device_name)
                self.publish_device_error_once(msg)
                with self.capture_state_lock:
                    self.recording = False

    def stop(self) -> None:
        with self.capture_state_lock:
            was_recording = self.recording
            self.recording = False

        if was_recording:
            self.close_stream_resources()

    async def wait_deliveries_drained(self, timeout_s: float = 3.0) -> None:
        """Wait until chunk callbacks scheduled before ``stop`` have run on ``loop``."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        poll_s = 0.02
        while True:
            with self._inflight_lock:
                if self._inflight_deliveries <= 0:
                    return
            if loop.time() >= deadline:
                with self._inflight_lock:
                    remaining = self._inflight_deliveries
                self.logger.warning(
                    "Timed out after %.1fs waiting for %s in-flight audio deliveries to finish",
                    timeout_s,
                    remaining,
                )
                return
            await asyncio.sleep(poll_s)

    def is_recording(self) -> bool:
        with self.capture_state_lock:
            return self.recording
