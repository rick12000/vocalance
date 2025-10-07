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



