import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from vocalance.app.config.app_config import XASRConfig


def test_utterance_idle_follows_hypothesis_changes(xasr_stream_session):
    xasr_stream_session.stop()
    assert xasr_stream_session.utterance_idle_seconds() is None
    xasr_stream_session.maybe_silence_commit("hello")
    assert xasr_stream_session.utterance_idle_seconds() < 0.5
    xasr_stream_session.hypothesis_stable_since = time.monotonic() - 3.0
    assert xasr_stream_session.utterance_idle_seconds() >= 3.0
    xasr_stream_session.maybe_silence_commit("hello there")
    assert xasr_stream_session.utterance_idle_seconds() < 0.5


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


def test_xasr_engine_uses_greedy_search_without_bpe_vocab(stt_config, tmp_path):
    missing = tmp_path / "bpe.vocab"
    with patch("sherpa_onnx.OnlineRecognizer.from_transducer") as mock_factory, patch(
        "vocalance.app.services.dictation_flow.speech_recognition.xasr_engine.bpe_vocab_path",
        return_value=missing,
    ):
        mock_factory.return_value = Mock()
        from vocalance.app.services.dictation_flow.speech_recognition.xasr_engine import XASREngine

        engine = XASREngine(config=stt_config)
        assert engine.hotwords_enabled is False
        assert mock_factory.call_args.kwargs["decoding_method"] == "greedy_search"


def test_xasr_engine_uses_beam_search_when_bpe_vocab_exists(stt_config, tmp_path):
    vocab = tmp_path / "bpe.vocab"
    vocab.write_text("▁A 0\n", encoding="utf-8")
    with patch("sherpa_onnx.OnlineRecognizer.from_transducer") as mock_factory, patch(
        "vocalance.app.services.dictation_flow.speech_recognition.xasr_engine.bpe_vocab_path",
        return_value=vocab,
    ):
        mock_factory.return_value = Mock()
        from vocalance.app.services.dictation_flow.speech_recognition.xasr_engine import XASREngine

        engine = XASREngine(config=stt_config)
        assert engine.hotwords_enabled is True
        kwargs = mock_factory.call_args.kwargs
        assert kwargs["decoding_method"] == "modified_beam_search"
        assert kwargs["modeling_unit"] == "bpe"
        assert kwargs["bpe_vocab"] == str(vocab)
        assert kwargs["max_active_paths"] == 4
        assert kwargs["hotwords_score"] == 2.0


def test_create_session_passes_hotwords_only_when_beam_search_is_active(stt_config, tmp_path):
    vocab = tmp_path / "bpe.vocab"
    vocab.write_text("▁A 0\n", encoding="utf-8")
    stream = Mock()
    stream.input_finished = Mock()
    recognizer = Mock()
    recognizer.create_stream.return_value = stream
    recognizer.is_ready.return_value = False
    recognizer.get_result.return_value = ""
    with patch("sherpa_onnx.OnlineRecognizer.from_transducer", return_value=recognizer), patch(
        "vocalance.app.services.dictation_flow.speech_recognition.xasr_engine.bpe_vocab_path",
        return_value=vocab,
    ):
        from vocalance.app.services.dictation_flow.speech_recognition.xasr_engine import XASREngine

        engine = XASREngine(config=stt_config)
        loop = asyncio.new_event_loop()
        try:
            session = engine.create_session(loop=loop, on_committed=AsyncMock(), on_provisional=AsyncMock(), hotwords="API/JSON")
            recognizer.create_stream.assert_called_with("API/JSON")
            session.stop()
            engine.hotwords_enabled = False
            session = engine.create_session(loop=loop, on_committed=AsyncMock(), on_provisional=AsyncMock(), hotwords="API")
            recognizer.create_stream.assert_called_with()
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
