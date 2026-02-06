import logging
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

import sounddevice as sd

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.event_bus import EventBus
from vocalance.app.events.core_events import AudioDeviceErrorEvent


class AudioRecorder:
    """Simple continuous audio chunk recorder.

    Captures audio from microphone at a fixed chunk size (50ms base unit) and
    continuously streams chunks via callback. No VAD logic - just pure streaming.
    Downstream listeners handle VAD, buffering, and segment detection.

    This design decouples audio capture from processing, enabling multiple independent
    listeners with different parameters (command, dictation, sound recognition, etc).

    Attributes:
        chunk_size: Audio chunk size in samples (50ms base unit = 800 samples at 16kHz).
        sample_rate: Audio sample rate in Hz (default 16000).
        device: Audio input device ID (None = system default).
    """

    def __init__(
        self,
        app_config: GlobalAppConfig,
        event_bus: EventBus,
        on_audio_chunk: Optional[Callable[[bytes, float], None]] = None,
    ) -> None:
        """Initialize continuous audio recorder.

        Args:
            app_config: Global application configuration.
            event_bus: Event bus for publishing device errors.
            on_audio_chunk: Callback invoked for every audio chunk captured.
                          Signature: (audio_bytes: bytes, timestamp: float) -> None
        """
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.app_config = app_config
        self.event_bus = event_bus
        self.on_audio_chunk = on_audio_chunk

        # 30ms base unit at 16kHz = 480 samples (industry standard for VAD)
        self.sample_rate = app_config.audio.sample_rate
        self.chunk_size = int(self.sample_rate * 0.03)  # 30ms chunks - better latency/stability
        self.device = getattr(app_config.audio, "device", None)
        self._preferred_device = self.device
        self._device_error_shown = False  # Track if we've shown error dialog

        # Thread and stream state
        self._is_recording: bool = False
        self._is_active: bool = True
        self._thread: Optional[threading.Thread] = None
        self._stream: Optional[sd.InputStream] = None
        self._lock = threading.Lock()

        self.logger.debug(
            f"AudioRecorder initialized: chunk_size={self.chunk_size} samples (30ms), sample_rate={self.sample_rate}Hz"
        )

    def _create_stream(self) -> bool:
        """Create and start audio input stream.

        Returns:
            True if stream created successfully, False otherwise.
        """
        try:
            device = self._preferred_device

            self._stream = sd.InputStream(
                samplerate=self.sample_rate, blocksize=self.chunk_size, channels=1, dtype="int16", device=device
            )
            self._stream.start()

            # Verify stream is actually working with a test read
            time.sleep(0.05)  # Let device settle
            test_data, _ = self._stream.read(self.chunk_size)
            if test_data is None or len(test_data) == 0:
                raise RuntimeError("Stream created but not producing audio data")

            # Log success
            if device is None:
                self.logger.info("Using system default audio device")
            else:
                device_info = sd.query_devices(device)
                self.logger.info(f"Using audio device: {device_info['name']}")

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
        """Publish device error event for UI layer to handle."""
        if self._device_error_shown:
            return  # Don't spam events

        self._device_error_shown = True

        try:
            import asyncio

            event = AudioDeviceErrorEvent(error_message=error_message, device_id=self._preferred_device)

            # Publish event safely (handle both sync and async contexts)
            try:
                loop = asyncio.get_running_loop()
                asyncio.run_coroutine_threadsafe(self.event_bus.publish(event), loop)
            except RuntimeError:
                # No running loop - create one temporarily
                asyncio.run(self.event_bus.publish(event))

        except Exception as e:
            self.logger.error(f"Failed to publish device error event: {e}")

    def _recording_thread(self) -> None:
        """Main recording loop - continuously streams audio chunks.

        Reads audio frames at fixed intervals and invokes callback with raw bytes.
        No automatic reconnection - user must manually select a device if connection is lost.
        """
        # Create initial stream
        if not self._create_stream():
            self.logger.error("Failed to create initial audio stream")
            self._publish_device_error("Failed to initialize audio device")
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
                    # Paused - sleep and skip
                    time.sleep(0.1)
                    continue

                # Read one chunk (30ms worth of audio)
                start_time = time.time()

                # Capture stream reference locally to avoid race condition
                current_stream = self._stream
                if current_stream is None:
                    # Stream was closed by set_preferred_device - recreate
                    if not self._create_stream():
                        self.logger.error("Failed to recreate stream after device change")
                        self._publish_device_error("Failed to switch to selected audio device")
                        break
                    continue

                data, overflowed = current_stream.read(self.chunk_size)

                if overflowed:
                    self.logger.warning("Audio input buffer overflow")

                read_duration = time.time() - start_time
                timestamp = time.time()

                # Throttling: Protect against non-blocking/instant returns
                expected_duration = self.chunk_size / self.sample_rate
                if read_duration < (expected_duration * 0.1):
                    time.sleep(max(0, expected_duration - read_duration))

                # Convert to bytes and invoke callback
                if self.on_audio_chunk:
                    audio_bytes = data.tobytes()
                    self.on_audio_chunk(audio_bytes, timestamp)

            except (OSError, RuntimeError, sd.PortAudioError) as e:
                self.logger.error(f"Audio device error: {e}")
                self._cleanup_stream()
                self._publish_device_error(f"Audio device connection lost: {e}")
                break  # Stop recording - user must manually fix

            except Exception as e:
                self.logger.exception(f"Unexpected error in recording loop: {e}")
                self._cleanup_stream()
                break

    def _cleanup_stream(self) -> None:
        """Clean up audio stream resources safely."""
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
        """Start the recording thread and begin streaming audio chunks.

        Thread-safe - multiple calls are ignored if already recording.
        """
        with self._lock:
            if self._is_recording:
                return
            self._is_recording = True
            self._thread = threading.Thread(target=self._recording_thread, daemon=False)
            self._thread.start()

    def stop(self) -> None:
        """Stop the recording thread and clean up audio resources.

        Sets the stop flag and waits up to 5 seconds for thread termination.
        """
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
        """Pause/resume audio streaming without stopping the thread.

        When inactive, the recorder still runs but skips reading/processing audio.

        Args:
            active: True to enable streaming, False to pause.
        """
        with self._lock:
            self._is_active = active

    def is_recording(self) -> bool:
        """Check if the recording thread is currently running.

        Returns:
            True if recording thread is active, False otherwise.
        """
        with self._lock:
            return self._is_recording

    def is_active(self) -> bool:
        """Check if audio streaming is currently enabled.

        Returns:
            True if streaming is enabled, False if paused.
        """
        with self._lock:
            return self._is_active

    def test_device(self, device_id: Optional[int]) -> Tuple[bool, str]:
        """Test if a device can be opened successfully.

        Args:
            device_id: Device ID to test, or None for system default.

        Returns:
            Tuple of (success: bool, error_message: str). Empty string if successful.
        """
        try:
            # Try to create a test stream
            test_stream = sd.InputStream(
                samplerate=self.sample_rate, blocksize=self.chunk_size, channels=1, dtype="int16", device=device_id
            )
            test_stream.start()

            # Try a test read
            time.sleep(0.05)
            test_data, _ = test_stream.read(self.chunk_size)

            # Clean up
            test_stream.stop()
            test_stream.close()

            if test_data is None or len(test_data) == 0:
                return False, "Device opened but produced no audio data"

            return True, ""

        except sd.PortAudioError as e:
            return False, f"PortAudio error: {e}"
        except Exception as e:
            return False, f"Device error: {e}"

    @staticmethod
    def query_available_devices() -> List[Tuple[int, str, bool]]:
        """Query all available audio input devices.

        Returns:
            List of tuples (device_id, device_name, is_default) for all input devices.
        """
        try:
            devices = sd.query_devices()
            hostapis = sd.query_hostapis()

            # Find the default input device index safely
            try:
                default_input = sd.query_devices(kind="input")
                default_id = default_input.get("index") if default_input else None
            except Exception:
                default_id = None

            # Identify Windows MME API index (usually most stable on Windows)
            mme_api_index = next((i for i, api in enumerate(hostapis) if "MME" in api["name"]), -1)

            # Filter and deduplicate devices
            # Key: Device Name -> Value: (device_id, device_name, is_default)
            unique_devices: Dict[str, Tuple[int, str, bool]] = {}

            for idx, device in enumerate(devices):
                if device["max_input_channels"] <= 0:
                    continue

                name = device["name"]
                is_default = idx == default_id
                api_index = device.get("hostapi", -1)

                # Decision logic for duplicate names:
                # 1. Always prefer the system default device
                # 2. If no existing entry, accept current
                # 3. If existing is not default, and current is MME, replace (prefer MME)

                if is_default:
                    unique_devices[name] = (idx, name, True)
                elif name not in unique_devices:
                    unique_devices[name] = (idx, name, False)
                elif api_index == mme_api_index and not unique_devices[name][2]:
                    unique_devices[name] = (idx, name, False)

            # Sort by name for UI consistency
            result = list(unique_devices.values())
            result.sort(key=lambda x: x[1])

            return result

        except Exception as e:
            logging.getLogger("AudioRecorder").error(f"Error querying audio devices: {e}")
            return []

    def set_preferred_device(self, device_id: Optional[int]) -> bool:
        """Set preferred audio input device and trigger reconnection.

        Args:
            device_id: Device ID to use, or None for system default.

        Returns:
            True if device change initiated successfully.
        """
        with self._lock:
            if device_id == self._preferred_device:
                self.logger.debug(f"Device {device_id} already set as preferred")
                return True

            self.logger.info(f"Changing preferred device from {self._preferred_device} to {device_id}")
            self._preferred_device = device_id
            self.device = device_id
            self._device_error_shown = False  # Reset error flag when manually switching

            # Force reconnection by cleaning up current stream
            if self._stream and self._is_recording:
                self.logger.info("Forcing device reconnection...")
                try:
                    if hasattr(self._stream, "active") and self._stream.active:
                        self._stream.stop()
                    self._stream.close()
                except Exception as e:
                    self.logger.debug(f"Error during forced device switch cleanup: {e}")
                finally:
                    self._stream = None

            return True

    def on_device_updated(self, device_id: Optional[int]) -> None:
        """Handle device update from settings coordinator.

        Called when audio.device setting is changed through the UI.
        Thread-safe method that triggers device switch.

        Args:
            device_id: New device ID to use, or None for system default.
        """
        self.logger.info(f"Device update received from settings: {device_id}")
        self.set_preferred_device(device_id)
