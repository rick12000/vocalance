"""Unit tests for MoonshineSTT batch path (mocked native layer)."""

import numpy as np


def _pcm16_bytes(duration_s: float, sr: int = 16000) -> bytes:
    n = int(duration_s * sr)
    return np.zeros(n, dtype=np.int16).tobytes()


def test_recognize_returns_empty_on_empty_audio(moonshine_stt_instance):
    assert moonshine_stt_instance.recognize_sync(b"", sample_rate=16000) == ""


def test_recognize_returns_empty_on_short_audio(moonshine_stt_instance):
    audio_bytes = _pcm16_bytes(0.05)
    assert moonshine_stt_instance.recognize_sync(audio_bytes, sample_rate=16000) == ""


def test_recognize_processes_valid_audio(moonshine_stt_instance):
    audio_bytes = _pcm16_bytes(1.0)
    result = moonshine_stt_instance.recognize_sync(audio_bytes, sample_rate=16000)
    assert "test" in result.lower()
