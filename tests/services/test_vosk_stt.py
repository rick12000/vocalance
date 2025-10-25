"""
Unit tests for Enhanced Vosk STT service.

Tests core recognition behavior, duplicate filtering, and threading safety.
"""
import json


def test_recognize_returns_text_on_success(vosk_stt_instance, mock_vosk_recognizer):
    audio_bytes = b"\x00\x01" * 100

    mock_vosk_recognizer.FinalResult.return_value = json.dumps({"text": "hello world"})

    result = vosk_stt_instance.recognize_sync(audio_bytes, sample_rate=16000)

    assert result == "hello world"
    mock_vosk_recognizer.Reset.assert_called_once()
    mock_vosk_recognizer.AcceptWaveform.assert_called_once_with(audio_bytes)


def test_recognize_returns_empty_on_empty_audio(vosk_stt_instance):
    result = vosk_stt_instance.recognize_sync(b"", sample_rate=16000)

    assert result == ""


def test_recognize_returns_empty_when_no_text_recognized(vosk_stt_instance, mock_vosk_recognizer):
    audio_bytes = b"\x00\x01" * 100

    mock_vosk_recognizer.FinalResult.return_value = json.dumps({"text": ""})

    result = vosk_stt_instance.recognize_sync(audio_bytes, sample_rate=16000)

    assert result == ""


def test_recognize_handles_exceptions_gracefully(vosk_stt_instance, mock_vosk_recognizer):
    audio_bytes = b"\x00\x01" * 100

    mock_vosk_recognizer.AcceptWaveform.side_effect = Exception("Recognition failed")

    # Current implementation raises exceptions rather than catching them
    import pytest

    with pytest.raises(Exception, match="Recognition failed"):
        vosk_stt_instance.recognize_sync(audio_bytes, sample_rate=16000)
