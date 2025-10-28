import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
from pydantic import BaseModel


class MockAudioConfig(BaseModel):
    sample_rate: int = 16000
    chunk_size: int = 960
    silence_timeout: float = 1.0
    energy_threshold: float = 0.002
    max_duration: float = 10.0


class SampleRecorder:
    def __init__(self, config: Optional[MockAudioConfig] = None):
        self.config = config or MockAudioConfig()
        self._stream: Optional[sd.InputStream] = None
        self.silence_threshold = self.config.energy_threshold * 0.35

    def _calculate_energy(self, audio_chunk: np.ndarray) -> float:
        if audio_chunk.dtype == np.int16:
            return np.sqrt(np.mean((audio_chunk.astype(np.float32) / 32768.0) ** 2))
        return np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2))

    def record_audio_bytes(self) -> bytes:
        with sd.InputStream(
            samplerate=self.config.sample_rate, blocksize=self.config.chunk_size, channels=1, dtype="int16"
        ) as stream:
            # Wait for speech detection
            speech_detected = False
            while not speech_detected:
                data, _ = stream.read(self.config.chunk_size)
                energy = self._calculate_energy(data)
                if energy > self.config.energy_threshold:
                    speech_detected = True
                    audio_buffer = [data]
                    break

            # Collect audio until silence
            silence_start = None
            recording_start = time.time()

            while True:
                data, _ = stream.read(self.config.chunk_size)
                energy = self._calculate_energy(data)
                audio_buffer.append(data)

                if energy < self.silence_threshold:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > self.config.silence_timeout:
                        break
                else:
                    silence_start = None

                if time.time() - recording_start > self.config.max_duration:
                    break

        return np.concatenate(audio_buffer).tobytes()

    def record_and_save(self, output_path: str) -> str:
        audio_bytes = self.record_audio_bytes()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as f:
            f.write(audio_bytes)

        return output_path


def main():
    # Configuration - override these values as needed
    output_dir = "recorded_samples"
    sample_rate = 16000
    silence_timeout = 1.0
    max_duration = 10.0

    # Generate timestamp-based filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"recorded_sample_{timestamp}.bytes"

    config = MockAudioConfig(sample_rate=sample_rate, silence_timeout=silence_timeout, max_duration=max_duration)

    recorder = SampleRecorder(config)
    output_path = os.path.join(output_dir, filename)

    recorder.record_and_save(output_path)


if __name__ == "__main__":
    main()
