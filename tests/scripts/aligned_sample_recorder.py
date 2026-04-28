import argparse
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import sounddevice as sd
from numpy.typing import NDArray

from vocalance.app.config.app_config import GlobalAppConfig

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


class AlignedSampleRecorder:
    """Standalone recorder that mirrors Vocalance's capture parameters.

    Uses ``sounddevice`` directly (no event bus, no service graph) so the
    script can run without bringing up the full application. Each captured
    PCM buffer is written verbatim to disk for offline analysis.
    """

    def __init__(self, mode: str, output_dir: str) -> None:
        self.mode = mode
        self.output_dir = output_dir
        self.segment_count = 0
        self.config = GlobalAppConfig()
        self.running = True

        self.sample_rate = int(self.config.audio.sample_rate)
        self.chunk_size = int(self.sample_rate * float(self.config.audio.capture_chunk_duration_seconds))
        self._stream: Optional[sd.InputStream] = None
        self._lock = threading.Lock()

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__)

    def _portaudio_callback(self, indata: NDArray[np.int16], frames: int, time_info: Any, status: Optional[Any]) -> None:
        if status:
            self.logger.debug("Input stream status: %s", status)
        with self._lock:
            if not self.running:
                return
            seq = self.segment_count
            self.segment_count += 1

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"{self.mode}_segment_{seq:03d}_{timestamp}.bytes"
        filepath = os.path.join(self.output_dir, filename)

        pcm_bytes = indata.tobytes()
        with open(filepath, "wb") as f:
            f.write(pcm_bytes)

        duration = len(pcm_bytes) / (self.sample_rate * 2)
        self.logger.info(f"Saved segment {seq}: {filename} ({duration:.2f}s, {len(pcm_bytes)} bytes)")

    def start(self) -> None:
        self.logger.info(f"Starting {self.mode} mode recorder")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info("Press Ctrl+C to stop recording")

        if self.mode == "command":
            self.logger.info(
                f"energy_threshold={self.config.vad.command_energy_threshold}, "
                f"silent_chunks_for_end={self.config.vad.command_silent_chunks_for_end} "
                f"(~{self.config.vad.command_silent_chunks_for_end * 50}ms)"
            )
        else:
            self.logger.info(
                "dictation label: continuous raw chunks only (in-app dictation uses Moonshine streaming, not VAD segments)"
            )

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                channels=1,
                dtype="int16",
                device=None,
                callback=self._portaudio_callback,
            )
            self._stream.start()
        except Exception as e:
            self.logger.error("Failed to open audio input stream: %s", e)
            return

        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self._shutdown()

    def _signal_handler(self, signum: int, frame: Any) -> None:
        self.logger.info("Interrupt received, stopping recorder...")
        with self._lock:
            self.running = False

    def _shutdown(self) -> None:
        with self._lock:
            self.running = False
        if self._stream is not None:
            try:
                if self._stream.active:
                    self._stream.stop()
                self._stream.close()
            except Exception as e:
                self.logger.warning("Error while closing audio stream: %s", e)
            self._stream = None
        self.logger.info(f"Recording stopped. Total segments saved: {self.segment_count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Record audio using Vocalance capture parameters")
    parser.add_argument(
        "--mode",
        choices=["command", "dictation"],
        default="command",
        help="Recording mode: command (optimized for speed) or dictation (optimized for accuracy)",
    )
    parser.add_argument(
        "--output-dir",
        default="recorded_samples/aligned",
        help="Output directory for saved segments (default: recorded_samples/aligned)",
    )

    args = parser.parse_args()

    recorder = AlignedSampleRecorder(mode=args.mode, output_dir=args.output_dir)
    recorder.start()


if __name__ == "__main__":
    main()
