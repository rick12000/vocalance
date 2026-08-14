import asyncio
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from vocalance.app.config.app_config import XASRConfig


def test_worker_thread_starts_on_init(xasr_stream_session):
    assert xasr_stream_session.worker.is_alive()
    xasr_stream_session.stop()


def test_worker_thread_terminates_after_stop(xasr_stream_session):
    assert xasr_stream_session.worker.is_alive()
    xasr_stream_session.stop()
    assert not xasr_stream_session.worker.is_alive()


def test_add_audio_after_stop_is_dropped(xasr_stream_session, mock_xasr_recognizer):
    recognizer, stream = mock_xasr_recognizer
    xasr_stream_session.stop()
    calls_before = stream.accept_waveform.call_count
    chunk = np.zeros(int(0.03 * 16000), dtype=np.int16).tobytes()
    xasr_stream_session.add_audio_pcm16(audio_bytes=chunk, sample_rate=16000)
    assert stream.accept_waveform.call_count == calls_before


def test_stop_calls_input_finished(xasr_stream_session, mock_xasr_recognizer):
    _, stream = mock_xasr_recognizer
    chunk = np.zeros(int(0.03 * 16000), dtype=np.int16).tobytes()
    xasr_stream_session.add_audio_pcm16(audio_bytes=chunk, sample_rate=16000)
    xasr_stream_session.stop()
    stream.input_finished.assert_called_once()


def test_audio_fed_to_stream(xasr_stream_session, mock_xasr_recognizer):
    _, stream = mock_xasr_recognizer
    chunk = np.zeros(int(0.03 * 16000), dtype=np.int16).tobytes()
    for _ in range(5):
        xasr_stream_session.add_audio_pcm16(audio_bytes=chunk, sample_rate=16000)
    xasr_stream_session.stop()
    assert stream.accept_waveform.call_count >= 1


def test_backpressure_drops_oldest_on_overflow():
    from vocalance.app.services.dictation_flow.speech_recognition.xasr_engine import XASRStreamSession

    stream = Mock()
    stream.accept_waveform = Mock()
    stream.input_finished = Mock()
    recognizer = Mock()
    recognizer.create_stream.return_value = stream
    recognizer.is_ready.return_value = False
    recognizer.decode_stream = Mock()
    recognizer.get_result.return_value = ""

    loop = asyncio.new_event_loop()
    try:
        session = XASRStreamSession(
            recognizer=recognizer,
            loop=loop,
            on_committed=AsyncMock(),
            on_provisional=AsyncMock(),
            xasr_config=XASRConfig(),
        )
        session.audio_queue.maxsize = 4
        chunk = np.zeros(int(0.03 * 16000), dtype=np.int16).tobytes()
        for _ in range(200):
            session.add_audio_pcm16(audio_bytes=chunk, sample_rate=16000)
        assert session.dropped_chunks > 0
        session.stop()
    finally:
        loop.close()


def test_xasr_engine_loads_with_mocked_recognizer(stt_config):
    with patch("sherpa_onnx.OnlineRecognizer.from_transducer") as mock_factory:
        mock_factory.return_value = Mock()
        from vocalance.app.services.dictation_flow.speech_recognition.xasr_engine import XASREngine

        engine = XASREngine(config=stt_config)
        assert engine.recognizer is not None
        mock_factory.assert_called_once()


def test_xasr_engine_create_session_raises_when_not_loaded(stt_config):
    with patch("sherpa_onnx.OnlineRecognizer.from_transducer") as mock_factory:
        mock_factory.return_value = Mock()
        from vocalance.app.services.dictation_flow.speech_recognition.xasr_engine import XASREngine

        engine = XASREngine(config=stt_config)
        engine.recognizer = None
        loop = asyncio.new_event_loop()
        try:
            with pytest.raises(RuntimeError):
                engine.create_session(loop=loop, on_committed=AsyncMock(), on_provisional=AsyncMock())
        finally:
            loop.close()
