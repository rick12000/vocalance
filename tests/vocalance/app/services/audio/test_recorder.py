import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from vocalance.app.services.audio.recorder import AudioRecorder


class FakeLoop:
    """Minimal loop surface so unit tests exercise ``call_soon_threadsafe`` without a running asyncio loop."""

    def is_running(self) -> bool:
        return True

    def call_soon_threadsafe(self, fn, *args) -> None:
        if args:
            fn(*args)
        else:
            fn()

    def create_task(self, coro):
        asyncio.run(coro)
        return Mock()


@pytest.fixture
def mock_app_config():
    config = Mock()
    config.audio = Mock()
    config.audio.sample_rate = 16000
    config.audio.capture_chunk_duration_seconds = 0.03
    messages = Mock()
    messages.message_for_launch_device = Mock(return_value="device error message")
    config.audio.device_capture_messages = messages
    return config


@pytest.fixture
def mock_callback():
    return Mock()


@pytest.fixture
def fake_loop():
    return FakeLoop()


@pytest.fixture
def mock_event_bus():
    bus = Mock()
    bus.publish = AsyncMock(return_value=None)
    return bus


def test_initialization_basic(mock_app_config, mock_callback, fake_loop, mock_event_bus):
    recorder = AudioRecorder(
        app_config=mock_app_config,
        loop=fake_loop,
        event_bus=mock_event_bus,
        on_audio_chunk=mock_callback,
    )

    assert recorder.sample_rate == 16000
    assert recorder.on_audio_chunk == mock_callback
    assert not recorder.is_recording()


def test_start_recording_changes_state(mock_app_config, mock_callback, fake_loop, mock_event_bus):
    with patch("vocalance.app.services.audio.recorder.sd.InputStream"):
        recorder = AudioRecorder(
            app_config=mock_app_config,
            loop=fake_loop,
            event_bus=mock_event_bus,
            on_audio_chunk=mock_callback,
        )

    recorder.start()
    time.sleep(0.1)

    assert recorder.is_recording()

    recorder.stop()


def test_stop_recording_changes_state(mock_app_config, mock_callback, fake_loop, mock_event_bus):
    with patch("vocalance.app.services.audio.recorder.sd.InputStream"):
        recorder = AudioRecorder(
            app_config=mock_app_config,
            loop=fake_loop,
            event_bus=mock_event_bus,
            on_audio_chunk=mock_callback,
        )

    recorder.start()
    time.sleep(0.1)
    assert recorder.is_recording()

    recorder.stop()
    assert not recorder.is_recording()


def test_multiple_start_stop_safe(mock_app_config, mock_callback, fake_loop, mock_event_bus):
    with patch("vocalance.app.services.audio.recorder.sd.InputStream"):
        recorder = AudioRecorder(
            app_config=mock_app_config,
            loop=fake_loop,
            event_bus=mock_event_bus,
            on_audio_chunk=mock_callback,
        )

    recorder.start()
    recorder.start()
    assert recorder.is_recording()

    recorder.stop()
    recorder.stop()
    assert not recorder.is_recording()


def test_callback_invoked_with_audio_chunks(mock_app_config, mock_callback, fake_loop, mock_event_bus):
    mock_stream = Mock()
    mock_audio_data = np.random.randint(-1000, 1000, size=(480, 1), dtype=np.int16)
    mock_stream.read.return_value = (mock_audio_data, None)
    mock_stream.active = True

    with patch("vocalance.app.services.audio.recorder.sd.InputStream", return_value=mock_stream):
        recorder = AudioRecorder(
            app_config=mock_app_config,
            loop=fake_loop,
            event_bus=mock_event_bus,
            on_audio_chunk=mock_callback,
        )

        recorder.start()
        recorder.portaudio_callback(mock_audio_data, 480, None, None)
        time.sleep(0.2)
        recorder.stop()

    assert mock_callback.call_count > 0

    call_args = mock_callback.call_args_list[0]
    audio_bytes, timestamp = call_args[0]

    assert isinstance(audio_bytes, bytes)
    assert isinstance(timestamp, float)
    assert len(audio_bytes) == 480 * 2


def test_callback_receives_audio_data(mock_app_config, mock_callback, fake_loop, mock_event_bus):
    mock_stream = Mock()
    mock_audio_data = np.random.randint(-1000, 1000, size=(800, 1), dtype=np.int16)
    mock_stream.read.return_value = (mock_audio_data, None)
    mock_stream.active = True

    with patch("vocalance.app.services.audio.recorder.sd.InputStream", return_value=mock_stream):
        recorder = AudioRecorder(
            app_config=mock_app_config,
            loop=fake_loop,
            event_bus=mock_event_bus,
            on_audio_chunk=mock_callback,
        )

        recorder.start()
        recorder.portaudio_callback(mock_audio_data, 480, None, None)
        time.sleep(0.2)
        recorder.stop()

    assert mock_callback.call_count > 0

    audio_bytes, timestamp = mock_callback.call_args[0]
    assert isinstance(audio_bytes, bytes)
    assert isinstance(timestamp, float)


def test_create_stream_handles_errors(mock_app_config, mock_callback, fake_loop, mock_event_bus):
    with patch("vocalance.app.services.audio.recorder.sd.InputStream") as mock_input_stream:
        mock_input_stream.side_effect = OSError("Device busy")

        recorder = AudioRecorder(
            app_config=mock_app_config,
            loop=fake_loop,
            event_bus=mock_event_bus,
            on_audio_chunk=mock_callback,
        )
        result = recorder.open_input_stream()

        assert result is False
