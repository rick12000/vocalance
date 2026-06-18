import json

import pytest


@pytest.mark.parametrize("recognized_text", ["hello world", "copy that", ""])
def test_recognize_sync_returns_final_result_text(vosk_engine_instance, mock_vosk_recognizer, recognized_text):
    mock_vosk_recognizer.FinalResult.return_value = json.dumps({"text": recognized_text})

    result = vosk_engine_instance.recognize_sync(b"\x00\x01" * 100, sample_rate=16000)

    assert result == recognized_text


def test_recognize_sync_defaults_to_empty_without_text_key(vosk_engine_instance, mock_vosk_recognizer):
    mock_vosk_recognizer.FinalResult.return_value = json.dumps({})

    result = vosk_engine_instance.recognize_sync(b"\x00\x01" * 100, sample_rate=16000)

    assert result == ""


def test_recognize_sync_skips_recognizer_on_empty_audio(vosk_engine_instance, mock_vosk_recognizer):
    result = vosk_engine_instance.recognize_sync(b"", sample_rate=16000)

    assert result == ""
    mock_vosk_recognizer.AcceptWaveform.assert_not_called()
