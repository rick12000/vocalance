"""
Records audio from microphone and saves as a raw .wav file.
Automatically stops recording when silence is detected or max duration is reached.

This simulates the raw audio input that would be fed into the AudioRecorder,
rather than the processed output bytes it generates. Useful for unit testing
the recorder with real audio data.
"""

import sounddevice as sd
import numpy as np
import time
import os
from pathlib import Path
from datetime import datetime
from typing import Optional
import wave
from pydantic import BaseModel


class AudioConfig(BaseModel):
    sample_rate: int = 16000
    chunk_size: int = 960
    silence_timeout: float = 1.0
    energy_threshold: float = 0.002
    max_duration: float = 10.0
    pre_roll_chunks: int = 5  # Number of chunks to keep before speech detection


class AudioRecorder:
    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        self._stream: Optional[sd.InputStream] = None
        self.silence_threshold = self.config.energy_threshold * 0.35

    def _calculate_energy(self, audio_chunk: np.ndarray) -> float:
        if audio_chunk.dtype == np.int16:
            return np.sqrt(np.mean((audio_chunk.astype(np.float32) / 32768.0) ** 2))
        return np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2))

    def record_audio_to_wav(self, output_path: str) -> str:
        """
        Record audio and save as .wav file, simulating raw input to AudioRecorder.
        This captures the exact audio stream that would be processed by the recorder.
        """
        audio_frames = []

        with sd.InputStream(
            samplerate=self.config.sample_rate,
            blocksize=self.config.chunk_size,
            channels=1,
            dtype='int16'
        ) as stream:
            # Wait for speech detection with pre-roll buffer
            pre_roll_buffer = []
            speech_detected = False

            while not speech_detected:
                data, _ = stream.read(self.config.chunk_size)
                energy = self._calculate_energy(data)

                # Maintain pre-roll buffer
                pre_roll_buffer.append(data)
                if len(pre_roll_buffer) > self.config.pre_roll_chunks:
                    pre_roll_buffer.pop(0)

                if energy > self.config.energy_threshold:
                    speech_detected = True
                    # Include pre-roll buffer in final audio
                    audio_frames.extend(pre_roll_buffer)
                    break

            # Collect audio until silence
            silence_start = None
            recording_start = time.time()

            while True:
                data, _ = stream.read(self.config.chunk_size)
                energy = self._calculate_energy(data)
                audio_frames.append(data)

                if energy < self.silence_threshold:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > self.config.silence_timeout:
                        break
                else:
                    silence_start = None

                if time.time() - recording_start > self.config.max_duration:
                    break

        # Save as .wav file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Concatenate all frames
        audio_data = np.concatenate(audio_frames)

        # Write .wav file
        with wave.open(output_path, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.config.sample_rate)
            wav_file.writeframes(audio_data.tobytes())

        duration = len(audio_data) / self.config.sample_rate
        print(f"Saved {duration:.2f}s of audio to {output_path}")
        return output_path


def main():
    # Configuration - override these values as needed
    output_dir = 'recorded_samples'
    sample_rate = 16000
    silence_timeout = 1.0
    max_duration = 10.0

    # Generate timestamp-based filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"recorded_audio_{timestamp}.wav"

    config = AudioConfig(
        sample_rate=sample_rate,
        silence_timeout=silence_timeout,
        max_duration=max_duration
    )

    recorder = AudioRecorder(config)
    output_path = os.path.join(output_dir, filename)

    print("Recording... Speak now!")
    recorder.record_audio_to_wav(output_path)
    print(f"Audio saved to: {output_path}")


if __name__ == '__main__':
    main()
