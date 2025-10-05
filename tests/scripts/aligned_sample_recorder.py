"""
Aligned Sample Recorder

Records audio using the same AudioRecorder class and configurations used in the main Iris application.
This generates realistic audio byte segments that match the exact format passed to stt_service.py,
making it ideal for unit testing downstream audio processing components.

The recorder runs continuously from script start until interrupted (Ctrl+C), saving each audio
segment as a separate .bytes file with timestamps.

Usage:
    python aligned_sample_recorder.py --mode command
    python aligned_sample_recorder.py --mode dictation

Arguments:
    --mode: Recording mode ('command' or 'dictation') - uses corresponding config from app_config
"""

import sys
import os
import signal
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from iris.app.config.app_config import GlobalAppConfig
from iris.app.services.audio.recorder import AudioRecorder


class AlignedSampleRecorder:
    def __init__(self, mode: str, output_dir: str):
        self.mode = mode
        self.output_dir = output_dir
        self.segment_count = 0
        self.config = GlobalAppConfig()
        self.recorder: Optional[AudioRecorder] = None
        self.running = True

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _on_audio_segment(self, audio_bytes: bytes):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        filename = f"{self.mode}_segment_{self.segment_count:03d}_{timestamp}.bytes"
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'wb') as f:
            f.write(audio_bytes)

        duration = len(audio_bytes) / (self.config.audio.sample_rate * 2)
        self.logger.info(f"Saved segment {self.segment_count}: {filename} ({duration:.2f}s, {len(audio_bytes)} bytes)")
        self.segment_count += 1

    def start(self):
        self.logger.info(f"Starting {self.mode} mode recorder using Iris AudioRecorder")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info("Press Ctrl+C to stop recording")

        if self.mode == "command":
            self.logger.info(f"Command mode config: chunk_size={self.config.audio.command_chunk_size}, "
                           f"energy_threshold={self.config.vad.command_energy_threshold}, "
                           f"silence_timeout={self.config.vad.command_silence_timeout}s")
        else:
            self.logger.info(f"Dictation mode config: chunk_size={self.config.audio.chunk_size}, "
                           f"energy_threshold={self.config.vad.dictation_energy_threshold}, "
                           f"silence_timeout={self.config.vad.dictation_silence_timeout}s")

        self.recorder = AudioRecorder(
            app_config=self.config,
            mode=self.mode,
            on_audio_segment=self._on_audio_segment
        )

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.recorder.start()

        try:
            while self.running:
                import time
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self._shutdown()

    def _signal_handler(self, signum, frame):
        self.logger.info("Interrupt received, stopping recorder...")
        self.running = False

    def _shutdown(self):
        if self.recorder:
            self.recorder.stop()
        self.logger.info(f"Recording stopped. Total segments saved: {self.segment_count}")


def main():
    parser = argparse.ArgumentParser(
        description='Record audio using Iris AudioRecorder configurations'
    )
    parser.add_argument(
        '--mode',
        choices=['command', 'dictation'],
        default='command',
        help='Recording mode: command (optimized for speed) or dictation (optimized for accuracy)'
    )
    parser.add_argument(
        '--output-dir',
        default='recorded_samples/aligned',
        help='Output directory for saved segments (default: recorded_samples/aligned)'
    )

    args = parser.parse_args()

    recorder = AlignedSampleRecorder(mode=args.mode, output_dir=args.output_dir)
    recorder.start()


if __name__ == '__main__':
    main()

