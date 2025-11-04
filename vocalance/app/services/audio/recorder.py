import logging
import threading
import time
from typing import Callable, Optional

import sounddevice as sd

from vocalance.app.config.app_config import GlobalAppConfig


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
        on_audio_chunk: Optional[Callable[[bytes, float], None]] = None,
    ) -> None:
        """Initialize continuous audio recorder.

        Args:
            app_config: Global application configuration.
            on_audio_chunk: Callback invoked for every audio chunk captured.
                          Signature: (audio_bytes: bytes, timestamp: float) -> None
        """
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.app_config = app_config
        self.on_audio_chunk = on_audio_chunk

        # 50ms base unit at 16kHz = 800 samples
        self.sample_rate = app_config.audio.sample_rate
        self.chunk_size = int(self.sample_rate * 0.05)  # 50ms chunks
        self.device = getattr(app_config.audio, "device", None)

        # Thread and stream state
        self._is_recording: bool = False
        self._is_active: bool = True
        self._thread: Optional[threading.Thread] = None
        self._stream: Optional[sd.InputStream] = None
        self._lock = threading.Lock()

        self.logger.debug(
            f"AudioRecorder initialized: chunk_size={self.chunk_size} samples (50ms), " f"sample_rate={self.sample_rate}Hz"
        )

    def _recording_thread(self) -> None:
        """Main recording loop - continuously streams audio chunks.

        Reads audio frames at fixed intervals and invokes callback with raw bytes.
        No buffering, no VAD, no state machine - pure streaming.
        """
        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate, blocksize=self.chunk_size, channels=1, dtype="int16", device=self.device
            )
            self._stream.start()
            self.logger.debug("Continuous audio streaming started")

            while True:
                with self._lock:
                    if not self._is_recording:
                        break
                    is_active = self._is_active

                if not is_active:
                    # Paused - sleep and skip
                    time.sleep(0.1)
                    continue

                try:
                    # Read one chunk (50ms worth of audio)
                    data, _ = self._stream.read(self.chunk_size)
                    timestamp = time.time()

                    # Convert to bytes and invoke callback
                    if self.on_audio_chunk:
                        audio_bytes = data.tobytes()
                        self.on_audio_chunk(audio_bytes, timestamp)

                except (OSError, RuntimeError) as e:
                    self.logger.error(f"Audio device error: {e}")
                    return
                except Exception as e:
                    self.logger.exception(f"Unexpected error in recording loop: {e}")
                    raise

        except Exception as e:
            self.logger.error(f"Recording thread error: {e}", exc_info=True)
        finally:
            self._cleanup_stream()

    def _cleanup_stream(self) -> None:
        """Clean up audio stream resources safely."""
        if self._stream:
            try:
                if hasattr(self._stream, "active") and self._stream.active:
                    self._stream.stop()
                    self.logger.debug("Audio stream stopped")

                self._stream.close()
                self.logger.debug("Audio stream closed")

            except Exception as e:
                self.logger.error(f"Error cleaning up audio stream: {e}", exc_info=True)
            finally:
                self._stream = None
                self.logger.info("Audio stream cleanup completed")

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
