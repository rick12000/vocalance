"""
Unit tests for Enhanced Vosk STT service.

Tests core recognition behavior, duplicate filtering, and threading safety.
"""
import json
from unittest.mock import Mock, MagicMock, patch
import pytest




def test_recognize_returns_text_on_success(vosk_stt_instance, mock_vosk_recognizer):
    audio_bytes = b'\x00\x01' * 100
    
    mock_vosk_recognizer.FinalResult.return_value = json.dumps({"text": "hello world"})
    
    result = vosk_stt_instance.recognize(audio_bytes, sample_rate=16000)
    
    assert result == "hello world"
    mock_vosk_recognizer.Reset.assert_called_once()
    mock_vosk_recognizer.AcceptWaveform.assert_called_once_with(audio_bytes)


def test_recognize_returns_empty_on_empty_audio(vosk_stt_instance):
    result = vosk_stt_instance.recognize(b'', sample_rate=16000)
    
    assert result == ""


def test_recognize_returns_empty_when_no_text_recognized(vosk_stt_instance, mock_vosk_recognizer):
    audio_bytes = b'\x00\x01' * 100
    
    mock_vosk_recognizer.FinalResult.return_value = json.dumps({"text": ""})
    
    result = vosk_stt_instance.recognize(audio_bytes, sample_rate=16000)
    
    assert result == ""


def test_recognize_filters_duplicates(vosk_stt_instance, mock_vosk_recognizer, mock_duplicate_filter):
    audio_bytes = b'\x00\x01' * 100
    
    mock_vosk_recognizer.FinalResult.return_value = json.dumps({"text": "hello"})
    mock_duplicate_filter.is_duplicate.return_value = True
    
    result = vosk_stt_instance.recognize(audio_bytes, sample_rate=16000)
    
    assert result == ""
    mock_duplicate_filter.is_duplicate.assert_called_once_with("hello")


def test_recognize_handles_exceptions_gracefully(vosk_stt_instance, mock_vosk_recognizer):
    audio_bytes = b'\x00\x01' * 100
    
    mock_vosk_recognizer.AcceptWaveform.side_effect = Exception("Recognition failed")
    
    result = vosk_stt_instance.recognize(audio_bytes, sample_rate=16000)
    
    assert result == ""


@pytest.mark.parametrize("audio_bytes,expected", [
    (b'', None),
    (None, None)
])
def test_recognize_streaming_handles_empty_audio(vosk_stt_instance, audio_bytes, expected):
    result = vosk_stt_instance.recognize_streaming(audio_bytes, is_final=False)
    
    assert result == expected


def test_recognize_streaming_returns_final_result(vosk_stt_instance, mock_vosk_recognizer):
    audio_bytes = b'\x00\x01' * 100
    
    mock_vosk_recognizer.AcceptWaveform.return_value = True
    mock_vosk_recognizer.Result.return_value = json.dumps({"text": "hello"})
    
    result = vosk_stt_instance.recognize_streaming(audio_bytes, is_final=False)
    
    assert result == "hello"


def test_recognize_streaming_returns_partial_result(vosk_stt_instance, mock_vosk_recognizer):
    audio_bytes = b'\x00\x01' * 100
    
    mock_vosk_recognizer.AcceptWaveform.return_value = False
    mock_vosk_recognizer.PartialResult.return_value = json.dumps({"partial": "hel"})
    
    result = vosk_stt_instance.recognize_streaming(audio_bytes, is_final=False)
    
    assert result == "hel"


@pytest.mark.parametrize("text,is_nonsense", [
    ("", True),
    (" ", True),
    ("a", True),
    ("hello", False),
    ("test test test", True),
    ("hello world", False),
    ("the the the", True)
])
def test_is_nonsense_detection(vosk_stt_instance, text, is_nonsense):
    result = vosk_stt_instance._is_nonsense(text)
    
    assert result == is_nonsense


def test_recognize_streaming_filters_nonsense(vosk_stt_instance, mock_vosk_recognizer):
    audio_bytes = b'\x00\x01' * 100
    
    mock_vosk_recognizer.AcceptWaveform.return_value = True
    mock_vosk_recognizer.Result.return_value = json.dumps({"text": "a"})
    
    result = vosk_stt_instance.recognize_streaming(audio_bytes, is_final=False)
    
    assert result is None


def test_recognize_streaming_handles_exceptions(vosk_stt_instance, mock_vosk_recognizer):
    audio_bytes = b'\x00\x01' * 100
    
    mock_vosk_recognizer.AcceptWaveform.side_effect = Exception("Streaming failed")
    
    result = vosk_stt_instance.recognize_streaming(audio_bytes, is_final=False)
    
    assert result is None

