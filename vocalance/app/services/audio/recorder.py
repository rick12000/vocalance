import asyncio
import logging
import threading
import time
from typing import Callable, Optional

import sounddevice as sd

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.event_bus import EventBus
from vocalance.app.events.core_events import AudioDeviceErrorEvent

_MIC_LOST_MESSAGE = (
    "The default microphone that was in use when Vocalance started is no longer available "
    "or could not be opened.\n\n"
    "Please reconnect your microphone or fix your system audio settings, then "
    "completely quit and restart Vocalance."
)


def _disconnect_user_message(launch_device_name: Optional[str]) -> str:
    if launch_device_name:
        return (
            f"The microphone that was in use when Vocalance started ({launch_device_name}) is no longer "
            "available or could not be opened.\n\n"
            "Please reconnect your microphone or fix your system audio settings, then "
            "completely quit and restart Vocalance."
        )
    return _MIC_LOST_MESSAGE


class AudioRecorder:
    """Continuous audio chunk recorder using the host default input device.

    Always uses PortAudio's default input device (``device=None``) — the system default
    at stream creation time. There is no in-app device switching; if capture fails
    after startup, the user is notified to restart the application.
    """

    def __init__(
        self,
        app_config: GlobalAppConfig,
        event_bus: EventBus,
        on_audio_chunk: Optional[Callable[[bytes, float], None]] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.app_config = app_config
        self.event_bus = event_bus
        self.on_audio_chunk = on_audio_chunk
        self.loop = loop or asyncio.get_running_loop()

        self.sample_rate = app_config.audio.sample_rate
        self.chunk_size = int(self.sample_rate * 0.03)
        self._device_error_shown = False
        self._launch_input_device_name: Optional[str] = None

        self._is_recording: bool = False
        self._is_active: bool = True
        self._stream: Optional[sd.InputStream] = None
        self._lock = threading.Lock()

        self.logger.debug(
            f"AudioRecorder initialized: chunk_size={self.chunk_size} samples (30ms), sample_rate={self.sample_rate}Hz"
        )

    def _record_launch_device_name(self) -> None:
        """Cache the default input device name after a successful stream open."""
        if self._launch_input_device_name is not None:
            return
        try:
            info = sd.query_devices(kind="input")
            name = info.get("name") if isinstance(info, dict) else None
            if name:
                self._launch_input_device_name = str(name)
                self.logger.info(f"Recording using default input at launch: {self._launch_input_device_name}")
        except Exception as e:
            self.logger.debug(f"Could not query default input device name: {e}")

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        if status:
            self.logger.warning(f"Audio stream status: {status}")

        with self._lock:
            if not self._is_recording or not self._is_active:
                return

        if self.on_audio_chunk:
            audio_bytes = indata.tobytes()
            timestamp = time.time()
            self.loop.call_soon_threadsafe(self.on_audio_chunk, audio_bytes, timestamp)

    def _create_stream(self) -> bool:
        """Create and start input stream on the host default input device."""
        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                channels=1,
                dtype="int16",
                device=None,
                callback=self._audio_callback,
            )
            self._stream.start()

            self._record_launch_device_name()
            self.logger.info("Using host default input device for capture")
            return True

        except Exception as e:
            self.logger.error(f"Failed to create audio stream: {e}")
            self._cleanup_stream()
            return False

    def _publish_device_error(self, error_message: str) -> None:
        if self._device_error_shown:
            return

        self._device_error_shown = True

        async def do_publish():
            try:
                event = AudioDeviceErrorEvent(error_message=error_message)
                await self.event_bus.publish(event)
            except Exception as e:
                self.logger.error(f"Failed to publish device error event: {e}")

        try:
            self.loop.call_soon_threadsafe(lambda: asyncio.create_task(do_publish()))
        except RuntimeError:
            asyncio.run(do_publish())

    def _cleanup_stream(self) -> None:
        if self._stream:
            try:
                if hasattr(self._stream, "active") and self._stream.active:
                    self._stream.stop()
                self._stream.close()
            except Exception as e:
                self.logger.debug(f"Error cleaning up audio stream: {e}")
            finally:
                self._stream = None

    def start(self) -> None:
        with self._lock:
            if self._is_recording:
                return
            self._is_recording = True

        if not self._create_stream():
            self.logger.error("Failed to create initial audio stream")
            self._publish_device_error(_disconnect_user_message(self._launch_input_device_name))
            with self._lock:
                self._is_recording = False

    def stop(self) -> None:
        with self._lock:
            if not self._is_recording:
                return
            self._is_recording = False

        self._cleanup_stream()

    def set_active(self, active: bool) -> None:
        with self._lock:
            self._is_active = active

    def is_recording(self) -> bool:
        with self._lock:
            return self._is_recording

    def is_active(self) -> bool:
        with self._lock:
            return self._is_active
