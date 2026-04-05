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
    ) -> None:
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.app_config = app_config
        self.event_bus = event_bus
        self.on_audio_chunk = on_audio_chunk

        self.sample_rate = app_config.audio.sample_rate
        self.chunk_size = int(self.sample_rate * 0.03)
        self._device_error_shown = False
        self._launch_input_device_name: Optional[str] = None

        self._is_recording: bool = False
        self._is_active: bool = True
        self._thread: Optional[threading.Thread] = None
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

    def _create_stream(self) -> bool:
        """Create and start input stream on the host default input device."""
        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                channels=1,
                dtype="int16",
                device=None,
            )
            self._stream.start()

            time.sleep(0.05)
            test_data, _ = self._stream.read(self.chunk_size)
            if test_data is None or len(test_data) == 0:
                raise RuntimeError("Stream created but not producing audio data")

            self._record_launch_device_name()
            self.logger.info("Using host default input device for capture")
            return True

        except Exception as e:
            self.logger.error(f"Failed to create audio stream: {e}")
            if self._stream:
                try:
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
            return False

    def _publish_device_error(self, error_message: str) -> None:
        if self._device_error_shown:
            return

        self._device_error_shown = True

        try:
            event = AudioDeviceErrorEvent(error_message=error_message)
            coro = self.event_bus.publish(event)

            try:
                loop = asyncio.get_running_loop()
                asyncio.run_coroutine_threadsafe(coro, loop)
            except RuntimeError:
                asyncio.run(coro)

        except Exception as e:
            self.logger.error(f"Failed to publish device error event: {e}")

    def _recording_thread(self) -> None:
        if not self._create_stream():
            self.logger.error("Failed to create initial audio stream")
            self._publish_device_error(_disconnect_user_message(self._launch_input_device_name))
            return

        while True:
            with self._lock:
                if not self._is_recording:
                    break

            try:
                with self._lock:
                    if not self._is_recording:
                        break
                    is_active = self._is_active

                if not is_active:
                    time.sleep(0.1)
                    continue

                start_time = time.time()
                current_stream = self._stream
                if current_stream is None:
                    if not self._create_stream():
                        self.logger.error("Failed to recreate audio stream")
                        self._publish_device_error(_disconnect_user_message(self._launch_input_device_name))
                        break
                    continue

                data, overflowed = current_stream.read(self.chunk_size)

                if overflowed:
                    self.logger.warning("Audio input buffer overflow")

                read_duration = time.time() - start_time
                timestamp = time.time()

                expected_duration = self.chunk_size / self.sample_rate
                if read_duration < (expected_duration * 0.1):
                    time.sleep(max(0, expected_duration - read_duration))

                if self.on_audio_chunk:
                    audio_bytes = data.tobytes()
                    self.on_audio_chunk(audio_bytes, timestamp)

            except (OSError, RuntimeError, sd.PortAudioError) as e:
                self.logger.error(f"Audio device error: {e}")
                self._cleanup_stream()
                self._publish_device_error(_disconnect_user_message(self._launch_input_device_name))
                break

            except Exception as e:
                self.logger.exception(f"Unexpected error in recording loop: {e}")
                self._cleanup_stream()
                break

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
            self._thread = threading.Thread(target=self._recording_thread, daemon=False)
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            if not self._is_recording:
                return
            self._is_recording = False

        if self._thread:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                self.logger.error("Recording thread did not terminate after 5s timeout")

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
