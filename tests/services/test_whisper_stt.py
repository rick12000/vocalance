"""
Unit tests for Whisper STT service.

Tests core recognition behavior, audio preparation, and text normalization.
"""
from unittest.mock import Mock, patch

import numpy as np
import pytest


def test_prepare_audio_converts_bytes_to_float32(whisper_stt_instance):
    audio_bytes = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16).tobytes()

    result = whisper_stt_instance._prepare_audio(audio_bytes)

    assert result.dtype == np.float32
    assert len(result) == 5
    assert -1.0 <= result.min() <= result.max() <= 1.0


def test_extract_text_from_segments_combines_text(whisper_stt_instance):
    mock_segment1 = Mock()
    mock_segment1.text = "hello"
    mock_segment1.avg_logprob = -0.5

    mock_segment2 = Mock()
    mock_segment2.text = "world"
    mock_segment2.avg_logprob = -0.3

    text, confidence = whisper_stt_instance._extract_text_from_segments([mock_segment1, mock_segment2])

    assert text == "hello world"
    assert 0.0 <= confidence <= 1.0


def test_extract_text_from_segments_skips_empty_segments(whisper_stt_instance):
    mock_segment1 = Mock()
    mock_segment1.text = "hello"
    mock_segment1.avg_logprob = -0.5

    mock_segment2 = Mock()
    mock_segment2.text = "   "
    mock_segment2.avg_logprob = -0.3

    text, confidence = whisper_stt_instance._extract_text_from_segments([mock_segment1, mock_segment2])

    assert text == "hello"


def test_extract_text_from_segments_handles_missing_logprob(whisper_stt_instance):
    mock_segment = Mock()
    mock_segment.text = "hello"
    mock_segment.avg_logprob = None

    text, confidence = whisper_stt_instance._extract_text_from_segments([mock_segment])

    assert text == "hello"
    assert confidence == 0.8


def test_recognize_returns_empty_on_empty_audio(whisper_stt_instance):
    result = whisper_stt_instance.recognize(b"", sample_rate=16000)

    assert result == ""


def test_recognize_returns_empty_on_short_audio(whisper_stt_instance):
    audio_bytes = np.zeros(1000, dtype=np.int16).tobytes()

    result = whisper_stt_instance.recognize(audio_bytes, sample_rate=16000)

    assert result == ""


def test_recognize_processes_valid_audio(whisper_stt_instance, mock_whisper_model):
    audio_bytes = np.random.randint(-1000, 1000, 16000, dtype=np.int16).tobytes()

    mock_segment = Mock()
    mock_segment.text = "hello world"
    mock_segment.avg_logprob = -0.5

    mock_info = Mock()
    mock_whisper_model.transcribe.return_value = ([mock_segment], mock_info)

    result = whisper_stt_instance.recognize(audio_bytes, sample_rate=16000)

    assert result == "hello world"
    assert mock_whisper_model.transcribe.call_count >= 1


def test_recognize_filters_duplicates(whisper_stt_instance, mock_whisper_model, mock_duplicate_filter):
    audio_bytes = np.random.randint(-1000, 1000, 16000, dtype=np.int16).tobytes()

    mock_segment = Mock()
    mock_segment.text = "hello"
    mock_segment.avg_logprob = -0.5

    mock_info = Mock()
    mock_whisper_model.transcribe.return_value = ([mock_segment], mock_info)
    mock_duplicate_filter.is_duplicate.return_value = True

    result = whisper_stt_instance.recognize(audio_bytes, sample_rate=16000)

    assert result == ""


def test_recognize_handles_exceptions(whisper_stt_instance, mock_whisper_model):
    audio_bytes = np.random.randint(-1000, 1000, 16000, dtype=np.int16).tobytes()

    mock_whisper_model.transcribe.side_effect = Exception("Recognition failed")

    result = whisper_stt_instance.recognize(audio_bytes, sample_rate=16000)

    assert result == ""


@pytest.mark.parametrize(
    "input_text,expected_output",
    [
        ("", ""),
        ("   ", ""),
        ("hello world", "hello world"),
        ("hello  world", "hello world"),
        ("um hello world", "hello world"),
        ("hello world uh", "hello world"),
        ("hello hello world", "hello world"),
        ("the the cat sat", "the cat sat"),
    ],
)
def test_normalize_text(whisper_stt_instance, input_text, expected_output):
    result = whisper_stt_instance._normalize_text(input_text)

    assert result == expected_output


def test_get_transcription_options_adjusts_for_short_audio(whisper_stt_instance):
    options = whisper_stt_instance._get_transcription_options(audio_duration=3.0)

    assert options["beam_size"] < whisper_stt_instance._beam_size
    assert options["language"] == "en"
    assert "vad_filter" in options
    assert options["vad_filter"] is True


def test_get_transcription_options_uses_full_beam_for_long_audio(whisper_stt_instance):
    options = whisper_stt_instance._get_transcription_options(audio_duration=10.0)

    assert options["beam_size"] == whisper_stt_instance._beam_size
    assert options["language"] == "en"


def test_recognize_warns_on_sample_rate_mismatch(whisper_stt_instance, mock_whisper_model):
    audio_bytes = np.random.randint(-1000, 1000, 16000, dtype=np.int16).tobytes()

    mock_segment = Mock()
    mock_segment.text = "test"
    mock_segment.avg_logprob = -0.5

    mock_info = Mock()
    mock_whisper_model.transcribe.return_value = ([mock_segment], mock_info)

    with patch("vocalance.app.services.audio.whisper_stt.logger") as mock_logger:
        result = whisper_stt_instance.recognize(audio_bytes, sample_rate=8000)

        mock_logger.warning.assert_called_once()
        assert result == "test"
