"""
Integration test for complete audio flow from wav input to text recognition.

This test simulates the entire audio processing pipeline:
1. Load wav file audio samples (simulating microphone input)
2. Process through the event-driven audio service flow
3. Verify STT recognition accuracy
4. Measure end-to-end latency performance
"""
import asyncio
import time
from typing import Optional, Tuple

import numpy as np
import pytest

from iris.app.events.core_events import CommandAudioSegmentReadyEvent, CommandTextRecognizedEvent


class RecognitionCapture:
    """Helper class to capture recognition results from events."""

    def __init__(self):
        self.recognized_text: Optional[str] = None
        self.processing_time_ms: Optional[float] = None
        self.engine: Optional[str] = None
        self.received_event = asyncio.Event()

    async def capture_handler(self, event: CommandTextRecognizedEvent):
        """Event handler to capture recognition results."""
        self.recognized_text = event.text
        self.processing_time_ms = event.processing_time_ms
        self.engine = event.engine
        self.received_event.set()

    async def wait_for_result(self, timeout: float = 5.0) -> bool:
        """Wait for recognition result with timeout."""
        try:
            await asyncio.wait_for(self.received_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def reset(self):
        """Reset for next capture."""
        self.recognized_text = None
        self.processing_time_ms = None
        self.engine = None
        self.received_event.clear()


def _convert_audio_to_bytes(audio_float: np.ndarray, target_sample_rate: int, current_sample_rate: int) -> bytes:
    """
    Convert float audio array to int16 bytes format matching the recorder output.

    Args:
        audio_float: Audio as float array (typically -1.0 to 1.0)
        target_sample_rate: Target sample rate for conversion
        current_sample_rate: Current sample rate of the audio

    Returns:
        Audio as int16 bytes
    """
    import librosa

    if current_sample_rate != target_sample_rate:
        audio_float = librosa.resample(audio_float, orig_sr=current_sample_rate, target_sr=target_sample_rate)

    audio_int16 = (audio_float * 32768.0).astype(np.int16)
    return audio_int16.tobytes()


async def _simulate_audio_flow(event_bus, stt_service, audio_bytes: bytes, sample_rate: int) -> Tuple[Optional[str], float]:
    """
    Simulate the complete audio flow from input to recognition.

    Args:
        event_bus: Event bus for publishing/subscribing
        stt_service: STT service instance
        audio_bytes: Raw audio bytes (int16 format)
        sample_rate: Sample rate of the audio

    Returns:
        Tuple of (recognized_text, end_to_end_latency_ms)
    """
    capture = RecognitionCapture()
    event_bus.subscribe(CommandTextRecognizedEvent, capture.capture_handler)

    flow_start = time.time()

    audio_event = CommandAudioSegmentReadyEvent(audio_bytes=audio_bytes, sample_rate=sample_rate)

    await event_bus.publish(audio_event)

    result_received = await capture.wait_for_result(timeout=2.0)

    flow_end = time.time()
    end_to_end_latency_ms = (flow_end - flow_start) * 1000

    if not result_received:
        return None, end_to_end_latency_ms

    return capture.recognized_text, end_to_end_latency_ms


@pytest.mark.asyncio
async def test_audio_flow_recognition_accuracy(event_bus, stt_service, audio_flow_samples, app_config):
    """
    Test that audio flow correctly recognizes expected text from wav files.

    Verifies:
    - Each wav file is processed through the complete flow
    - Recognized text matches expected text from filename
    """
    if not audio_flow_samples:
        pytest.skip("No audio flow samples found")

    results = []

    for expected_text, (audio_data, sample_rate, filename) in audio_flow_samples.items():
        audio_bytes = _convert_audio_to_bytes(
            audio_data, target_sample_rate=app_config.audio.sample_rate, current_sample_rate=sample_rate
        )

        recognized_text, latency_ms = await _simulate_audio_flow(event_bus, stt_service, audio_bytes, app_config.audio.sample_rate)

        results.append(
            {
                "filename": filename,
                "expected": expected_text,
                "recognized": recognized_text,
                "latency_ms": latency_ms,
                "match": recognized_text == expected_text if recognized_text else False,
            }
        )

    for result in results:
        assert result["recognized"] is not None

        assert result["match"]


@pytest.mark.asyncio
async def test_audio_flow_latency_performance(event_bus, stt_service, audio_flow_samples, app_config):
    """
    Test end-to-end latency performance across multiple runs.

    Verifies:
    - Average latency for 'click' command is < 250ms
    - Average latency for 'right click' command is < 500ms

    Each sample is tested 10 times to get reliable average latency.
    """
    if not audio_flow_samples:
        pytest.skip("No audio flow samples found")

    latency_requirements = {"click": 250.0, "right click": 500.0}

    repetitions = 10
    latency_results = {}

    for expected_text, (audio_data, sample_rate, filename) in audio_flow_samples.items():
        if expected_text not in latency_requirements:
            continue

        audio_bytes = _convert_audio_to_bytes(
            audio_data, target_sample_rate=app_config.audio.sample_rate, current_sample_rate=sample_rate
        )

        latencies = []

        for rep in range(repetitions + 1):
            recognized_text, latency_ms = await _simulate_audio_flow(
                event_bus, stt_service, audio_bytes, app_config.audio.sample_rate
            )

            if recognized_text is not None and rep > 0:
                latencies.append(latency_ms)

            await asyncio.sleep(0.05)

        if not latencies:
            pytest.fail(f"No successful recognitions for {filename} across {repetitions} repetitions")

        avg_latency = np.mean(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
        std_latency = np.std(latencies)

        latency_results[expected_text] = {
            "avg": avg_latency,
            "min": min_latency,
            "max": max_latency,
            "std": std_latency,
            "requirement": latency_requirements[expected_text],
            "samples": len(latencies),
        }

    for expected_text, metrics in latency_results.items():
        assert metrics["avg"] < metrics["requirement"]


@pytest.mark.asyncio
async def test_audio_flow_end_to_end_integration(event_bus, stt_service, audio_flow_samples, app_config):
    """
    Combined test for recognition accuracy and performance characteristics.

    This test verifies the complete integration by:
    1. Processing all audio samples through the full flow
    2. Checking recognition accuracy for each sample
    3. Measuring and reporting latency statistics
    """
    if not audio_flow_samples:
        pytest.skip("No audio flow samples found")

    all_results = []

    for expected_text, (audio_data, sample_rate, filename) in audio_flow_samples.items():
        audio_bytes = _convert_audio_to_bytes(
            audio_data, target_sample_rate=app_config.audio.sample_rate, current_sample_rate=sample_rate
        )

        recognized_text, latency_ms = await _simulate_audio_flow(event_bus, stt_service, audio_bytes, app_config.audio.sample_rate)

        result = {
            "filename": filename,
            "expected": expected_text,
            "recognized": recognized_text,
            "latency_ms": latency_ms,
            "correct": recognized_text == expected_text if recognized_text else False,
        }

        all_results.append(result)

    accuracy = sum(1 for r in all_results if r["correct"]) / len(all_results)
    avg_latency = np.mean([r["latency_ms"] for r in all_results])

    assert accuracy == 1.0

    assert avg_latency < 1000.0
