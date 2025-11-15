import time
from unittest.mock import Mock, patch

import numpy as np
import pytest

from vocalance.app.services.audio.recorder import AudioRecorder


@pytest.fixture
def mock_app_config():
    config = Mock()
    config.audio = Mock()
    config.audio.sample_rate = 16000
    config.audio.device = None
    return config


@pytest.fixture
def mock_callback():
    return Mock()


def test_initialization_basic(mock_app_config, mock_callback):
    """Test basic recorder initialization."""
    recorder = AudioRecorder(app_config=mock_app_config, on_audio_chunk=mock_callback)

    assert recorder.sample_rate == 16000
    assert recorder.on_audio_chunk == mock_callback
    assert not recorder.is_recording()
    assert recorder.is_active()


def test_start_recording_changes_state(mock_app_config, mock_callback):
    with patch("vocalance.app.services.audio.recorder.sd.InputStream"):
        recorder = AudioRecorder(app_config=mock_app_config, on_audio_chunk=mock_callback)

        recorder.start()
        time.sleep(0.1)

        assert recorder.is_recording()

        recorder.stop()


def test_stop_recording_changes_state(mock_app_config, mock_callback):
    with patch("vocalance.app.services.audio.recorder.sd.InputStream"):
        recorder = AudioRecorder(app_config=mock_app_config, on_audio_chunk=mock_callback)

        recorder.start()
        time.sleep(0.1)
        assert recorder.is_recording()

        recorder.stop()
        assert not recorder.is_recording()


def test_multiple_start_stop_safe(mock_app_config, mock_callback):
    """Test that multiple start/stop calls are handled safely."""
    with patch("vocalance.app.services.audio.recorder.sd.InputStream"):
        recorder = AudioRecorder(app_config=mock_app_config, on_audio_chunk=mock_callback)

        # Multiple starts should be safe
        recorder.start()
        recorder.start()
        assert recorder.is_recording()

        # Multiple stops should be safe
        recorder.stop()
        recorder.stop()
        assert not recorder.is_recording()


def test_callback_invoked_with_audio_chunks(mock_app_config, mock_callback):
    mock_stream = Mock()
    mock_audio_data = np.random.randint(-1000, 1000, size=(800, 1), dtype=np.int16)
    mock_stream.read.return_value = (mock_audio_data, None)
    mock_stream.active = True

    with patch("vocalance.app.services.audio.recorder.sd.InputStream", return_value=mock_stream):
        recorder = AudioRecorder(app_config=mock_app_config, on_audio_chunk=mock_callback)

        recorder.start()
        time.sleep(0.2)
        recorder.stop()

        assert mock_callback.call_count > 0

        call_args = mock_callback.call_args_list[0]
        audio_bytes, timestamp = call_args[0]

        assert isinstance(audio_bytes, bytes)
        assert isinstance(timestamp, float)
        assert len(audio_bytes) == 800 * 2


def test_callback_receives_audio_data(mock_app_config, mock_callback):
    """Test that callback receives properly formatted audio data."""
    mock_stream = Mock()
    mock_audio_data = np.random.randint(-1000, 1000, size=(800, 1), dtype=np.int16)
    mock_stream.read.return_value = (mock_audio_data, None)
    mock_stream.active = True

    with patch("vocalance.app.services.audio.recorder.sd.InputStream", return_value=mock_stream):
        recorder = AudioRecorder(app_config=mock_app_config, on_audio_chunk=mock_callback)

        recorder.start()
        time.sleep(0.2)
        recorder.stop()

        assert mock_callback.call_count > 0

        # Verify callback receives bytes and timestamp
        audio_bytes, timestamp = mock_callback.call_args[0]
        assert isinstance(audio_bytes, bytes)
        assert isinstance(timestamp, float)
