import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf

from vocalance.app.config.app_config import GlobalAppConfig

# Add the project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


class AudioPreprocessor:
    """Standalone audio preprocessor that matches Vocalance training preprocessing exactly.

    This implements the same preprocessing pipeline as the AudioPreprocessor in
    streamlined_sound_recognizer.py, but as a standalone class to avoid dependency issues.
    """

    def __init__(self, config):
        """Initialize preprocessor with Vocalance sound recognition config."""
        self.target_sr = config.target_sample_rate
        self.silence_threshold = config.silence_threshold
        self.min_sound_duration = config.min_sound_duration
        self.max_sound_duration = config.max_sound_duration
        self.frame_length = config.frame_length
        self.hop_length = config.hop_length
        self.normalization_level = config.normalization_level
        # Flag to indicate if audio is already VAD-segmented (skip silence trimming)
        self.skip_silence_trimming = True  # VAD already segments properly

    def preprocess_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Essential preprocessing pipeline: resample, trim silence, normalize.

        Args:
            audio: Input audio numpy array.
            sr: Sample rate of input audio.

        Returns:
            Preprocessed audio array ready for embedding extraction.
        """
        if not isinstance(audio, np.ndarray):
            raise TypeError("Audio must be a numpy array")

        if len(audio) == 0:
            raise ValueError("Audio array is empty")

        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        if not isinstance(sr, (int, np.integer)) or sr <= 0:
            raise ValueError(f"Invalid sample rate: {sr}")

        if sr != self.target_sr:
            try:
                audio = librosa.resample(y=audio, orig_sr=sr, target_sr=self.target_sr)
            except Exception as e:
                raise ValueError(f"Failed to resample audio: {e}")

        # OPTIMIZATION: Skip silence trimming for VAD-segmented audio
        # The SoundAudioListener already performs energy-based onset/offset detection,
        # so additional trimming creates a double-filtering effect and may remove
        # important transient characteristics. Pre-roll buffer also captures onset.
        if not self.skip_silence_trimming:
            audio = self._trim_silence(audio)

        duration = len(audio) / self.target_sr

        # IMPROVED DURATION NORMALIZATION: Use symmetric padding and center cropping
        # This preserves sound characteristics better than left-aligned operations
        if duration < self.min_sound_duration:
            target_samples = int(self.min_sound_duration * self.target_sr)
            pad_total = target_samples - len(audio)
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            audio = np.pad(audio, (pad_left, pad_right), mode="constant", constant_values=0)
        elif duration > self.max_sound_duration:
            target_samples = int(self.max_sound_duration * self.target_sr)
            start_sample = (len(audio) - target_samples) // 2
            audio = audio[start_sample : start_sample + target_samples]

        # Apply amplitude normalization
        if self.normalization_level > 0:
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio * (self.normalization_level / max_val)

        return audio

    def _trim_silence(self, audio: np.ndarray) -> np.ndarray:
        """Trim silence from audio using RMS energy threshold."""
        # Calculate RMS energy in frames
        rms = librosa.feature.rms(y=audio, frame_length=self.frame_length, hop_length=self.hop_length)[0]

        # Find frames above threshold
        non_silent_frames = rms > self.silence_threshold

        if not np.any(non_silent_frames):
            return audio

        # Find start and end indices
        start_frame = np.argmax(non_silent_frames)
        end_frame = len(non_silent_frames) - np.argmax(non_silent_frames[::-1])

        # Convert to sample indices
        start_sample = start_frame * self.hop_length
        end_sample = min(end_frame * self.hop_length, len(audio))

        return audio[start_sample:end_sample]


class SoundRecognitionSampleRecorder:
    """Record audio samples that exactly match the preprocessing used in sound recognition training.

    This recorder mimics the exact workflow used by Vocalance for training new user sounds:
    1. Records audio at the target sample rate (16000 Hz)
    2. Applies identical preprocessing as the AudioPreprocessor
    3. Saves .wav files ready for integration testing

    The preprocessing includes:
    - Resampling to 16000 Hz (if needed)
    - Stereo to mono conversion
    - Duration normalization to 2.0 seconds
    - Amplitude normalization
    """

    def __init__(self, output_dir: str, sound_label: str, num_samples: int = 12):
        """Initialize the recorder with training parameters.

        Args:
            output_dir: Directory to save recorded .wav files
            sound_label: Label for the sound being recorded (used in filenames)
            num_samples: Number of samples to record (default: 12, matches training default)
        """
        self.output_dir = Path(output_dir)
        self.sound_label = sound_label
        self.num_samples = num_samples
        self.recorded_count = 0

        # Use Vocalance's configuration
        config = GlobalAppConfig()
        sound_config = config.sound_recognizer

        # Match exact training parameters
        self.sample_rate = sound_config.target_sample_rate  # 16000 Hz
        self.sample_duration_sec = sound_config.sample_duration_sec  # 2.0 seconds
        self.energy_threshold = sound_config.energy_threshold  # 0.001
        self.silence_threshold = sound_config.silence_threshold  # 0.005
        self.min_sound_duration = sound_config.min_sound_duration  # 0.1 seconds
        self.max_sound_duration = sound_config.max_sound_duration  # 2.0 seconds

        # Initialize preprocessor with exact same config as training
        self.preprocessor = AudioPreprocessor(config=sound_config)

        # Setup logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _calculate_energy(self, audio_chunk: np.ndarray) -> float:
        """Calculate RMS energy of audio chunk - matches training energy calculation."""
        if audio_chunk.dtype == np.int16:
            # Convert int16 to float32 in range [-1, 1]
            audio_float = audio_chunk.astype(np.float32) / 32768.0
        else:
            audio_float = audio_chunk.astype(np.float32)

        return np.sqrt(np.mean(audio_float**2))

    def _record_single_sample(self) -> Optional[np.ndarray]:
        """Record a single audio sample with VAD-like detection.

        Uses energy-based voice activity detection similar to how training samples
        are collected during the actual training process.

        Returns:
            Preprocessed audio array ready for embedding extraction, or None if failed
        """
        self.logger.info("Listening for sound... (make your sound now)")

        # Record with energy-based start detection
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=960,  # ~60ms chunks at 16kHz
        ) as stream:
            audio_buffer = []
            speech_started = False
            silence_start = None
            recording_start = time.time()

            while True:
                # Read audio chunk
                chunk, _ = stream.read(960)
                energy = self._calculate_energy(chunk)

                if not speech_started:
                    # Wait for speech to start
                    if energy > self.energy_threshold:
                        speech_started = True
                        audio_buffer = [chunk]  # Start recording with this chunk
                        self.logger.debug(".2f")
                    continue

                # Speech has started, collect audio
                audio_buffer.append(chunk)

                # Check for silence (end of sound)
                if energy < self.silence_threshold:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > 0.5:  # 500ms silence threshold
                        break
                else:
                    silence_start = None

                # Safety timeout
                if time.time() - recording_start > 5.0:  # Max 5 seconds
                    self.logger.warning("Recording timeout - using available audio")
                    break

        if not audio_buffer:
            self.logger.error("No audio detected")
            return None

        # Concatenate all chunks
        raw_audio = np.concatenate(audio_buffer)

        # Apply exact same preprocessing as used in training
        try:
            preprocessed_audio = self.preprocessor.preprocess_audio(audio=raw_audio, sr=self.sample_rate)
            self.logger.debug(f"Preprocessed audio: {len(preprocessed_audio)} samples " ".2f")
            return preprocessed_audio

        except Exception as e:
            self.logger.error(f"Preprocessing failed: {e}")
            return None

    def record_samples(self) -> List[str]:
        """Record the specified number of samples.

        Returns:
            List of paths to saved .wav files
        """
        saved_files = []

        self.logger.info(f"Recording {self.num_samples} samples for sound: '{self.sound_label}'")
        self.logger.info("Make your sound when prompted. Samples will be automatically detected.")
        self.logger.info("Press Ctrl+C to stop early")

        try:
            for i in range(self.num_samples):
                self.logger.info(f"\n--- Sample {i+1}/{self.num_samples} ---")

                # Record sample
                audio = self._record_single_sample()
                if audio is None:
                    self.logger.warning(f"Failed to record sample {i+1}, skipping")
                    continue

                # Generate filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                filename = f"{self.sound_label}_sample_{i+1:02d}_{timestamp}.wav"
                filepath = self.output_dir / filename

                # Save as .wav file
                sf.write(str(filepath), audio, self.sample_rate)
                saved_files.append(str(filepath))

                len(audio) / self.sample_rate
                self.logger.info(".2f" f"{len(audio)} samples")

                self.recorded_count += 1

                # Small delay between samples
                if i < self.num_samples - 1:
                    time.sleep(1.0)

        except KeyboardInterrupt:
            self.logger.info("Recording interrupted by user")

        self.logger.info(f"\nRecording complete: {len(saved_files)}/{self.num_samples} samples saved")
        return saved_files

    def generate_test_samples(self) -> List[str]:
        """Generate test .wav files with synthetic audio for testing purposes."""
        saved_files = []

        self.logger.info(f"Generating {self.num_samples} test samples for sound: '{self.sound_label}'")

        for i in range(self.num_samples):
            # Generate synthetic audio that mimics the characteristics of recorded samples
            # Create a 2-second audio clip at target sample rate
            duration_samples = int(self.sample_duration_sec * self.sample_rate)

            # Generate a simple test sound (combination of frequencies to simulate various sounds)
            t = np.linspace(0, self.sample_duration_sec, duration_samples, False)

            # Create a test signal with multiple frequency components
            freq1, freq2, freq3 = 440, 880, 1320  # A4, A5, A6
            amplitude = 0.3

            # Mix different waveforms to simulate various sound types
            test_audio = (
                amplitude * np.sin(2 * np.pi * freq1 * t)
                + amplitude * 0.5 * np.sin(2 * np.pi * freq2 * t)  # Sine wave
                + amplitude * 0.3 * np.sin(2 * np.pi * freq3 * t)  # Harmonic
                + amplitude * 0.2 * np.random.randn(len(t))  # Another harmonic  # Add some noise
            )

            # Normalize to prevent clipping
            test_audio = test_audio / np.max(np.abs(test_audio)) * 0.7

            # Apply the same preprocessing as real training
            try:
                preprocessed_audio = self.preprocessor.preprocess_audio(audio=test_audio.astype(np.float32), sr=self.sample_rate)

                # Generate filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                filename = f"{self.sound_label}_sample_{i+1:02d}_{timestamp}.wav"
                filepath = self.output_dir / filename

                # Save as .wav file
                sf.write(str(filepath), preprocessed_audio, self.sample_rate)
                saved_files.append(str(filepath))

                duration = len(preprocessed_audio) / self.sample_rate
                self.logger.info(
                    f"Generated test sample {i+1}: {filename} " f"({duration:.2f}s, {len(preprocessed_audio)} samples)"
                )

                self.recorded_count += 1

            except Exception as e:
                self.logger.error(f"Failed to generate test sample {i+1}: {e}")
                continue

        return saved_files

    def get_recording_stats(self) -> dict:
        """Get statistics about the recording session."""
        return {
            "sound_label": self.sound_label,
            "target_samples": self.num_samples,
            "recorded_samples": self.recorded_count,
            "sample_rate": self.sample_rate,
            "duration_seconds": self.sample_duration_sec,
            "output_directory": str(self.output_dir),
        }


def main():
    # Configuration - override these values as needed
    sound_label = "finger_snap"  # Label for the sound being recorded
    output_dir = "recorded_sound_samples"  # Output directory for .wav files
    num_samples = 12  # Number of samples to record (matches training default)
    use_test_generate = False  # Set to True to generate test files instead of recording from microphone

    # Generate timestamp-based filename pattern
    datetime.now().strftime("%Y%m%d_%H%M%S")

    recorder = SoundRecognitionSampleRecorder(output_dir=output_dir, sound_label=sound_label, num_samples=num_samples)

    # Show configuration
    GlobalAppConfig()
    mode = "Generate" if use_test_generate else "Record"
    print(f"Sound Recognition Sample {mode}er")
    print("=" * 40)
    print(f"Sound Label: {sound_label}")
    print(f"Samples to {mode.lower()}: {num_samples}")
    print(f"Sample Rate: {recorder.sample_rate} Hz")
    print(f"Sample Duration: {recorder.sample_duration_sec} seconds")
    if not use_test_generate:
        print(f"Energy Threshold: {recorder.energy_threshold}")
    print(f"Output Directory: {output_dir}")
    print()

    # Record or generate samples
    if use_test_generate:
        saved_files = recorder.generate_test_samples()
    else:
        saved_files = recorder.record_samples()

    # Show summary
    print("\nRecording Summary:")
    print(f"Successfully recorded: {len(saved_files)}/{num_samples} samples")
    print(f"Files saved to: {output_dir}/")
    print("\nSaved files:")
    for filepath in saved_files:
        print(f"  - {filepath}")

    # Show stats
    stats = recorder.get_recording_stats()
    print("\nRecording Parameters:")
    print(f"  Sample Rate: {stats['sample_rate']} Hz")
    print(f"  Duration: {stats['duration_seconds']} seconds")
    print("  Preprocessing: Identical to Vocalance training pipeline")


if __name__ == "__main__":
    main()
