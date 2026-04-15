import argparse
import asyncio
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.event_bus import EventBus
from vocalance.app.services.audio.recorder import AudioRecorder

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


class AlignedSampleRecorder:
    def __init__(self, mode: str, output_dir: str):
        self.mode = mode
        self.output_dir = output_dir
        self.segment_count = 0
        self.config = GlobalAppConfig()
        self.recorder: Optional[AudioRecorder] = None
        self.running = True
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._event_bus: Optional[EventBus] = None

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__)

    def _on_audio_segment(self, audio_bytes: bytes, _timestamp: float) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"{self.mode}_segment_{self.segment_count:03d}_{timestamp}.bytes"
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, "wb") as f:
            f.write(audio_bytes)

        duration = len(audio_bytes) / (self.config.audio.sample_rate * 2)
        self.logger.info(f"Saved segment {self.segment_count}: {filename} ({duration:.2f}s, {len(audio_bytes)} bytes)")
        self.segment_count += 1

    def _run_loop(self) -> None:
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def start(self) -> None:
        self.logger.info(f"Starting {self.mode} mode recorder using Vocalance AudioRecorder")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info("Press Ctrl+C to stop recording")

        if self.mode == "command":
            self.logger.info(
                f"energy_threshold={self.config.vad.command_energy_threshold}, "
                f"silent_chunks_for_end={self.config.vad.command_silent_chunks_for_end} (~{self.config.vad.command_silent_chunks_for_end * 50}ms)"
            )
        else:
            self.logger.info(
                "dictation label: continuous raw chunks only (in-app dictation uses Moonshine streaming, not VAD segments)"
            )

        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_loop, name="aligned-sample-asyncio", daemon=True)
        self._loop_thread.start()
        deadline = time.time() + 10.0
        while time.time() < deadline and not self._loop.is_running():
            time.sleep(0.01)

        self._event_bus = EventBus()
        self._event_bus.start()

        self.recorder = AudioRecorder(
            app_config=self.config,
            loop=self._loop,
            event_bus=self._event_bus,
            on_audio_chunk=self._on_audio_segment,
        )

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.recorder.start()

        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self._shutdown()

    def _signal_handler(self, signum, frame):
        self.logger.info("Interrupt received, stopping recorder...")
        self.running = False

    def _shutdown(self) -> None:
        if self.recorder:
            self.recorder.stop()
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=5.0)
        self.logger.info(f"Recording stopped. Total segments saved: {self.segment_count}")


def main():
    parser = argparse.ArgumentParser(description="Record audio using Vocalance AudioRecorder configurations")
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
