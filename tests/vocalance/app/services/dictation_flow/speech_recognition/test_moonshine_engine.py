import asyncio
from unittest.mock import AsyncMock, Mock

import numpy as np

from vocalance.app.config.app_config import MoonshineStreamingConfig


def pcm16_bytes(duration_s: float, sr: int = 16000) -> bytes:
    n = int(duration_s * sr)
    return np.zeros(n, dtype=np.int16).tobytes()


def test_recognize_returns_empty_on_empty_audio(moonshine_engine_instance):
    assert moonshine_engine_instance.recognize_sync(b"", sample_rate=16000) == ""


def test_recognize_returns_empty_on_short_audio(moonshine_engine_instance):
    audio_bytes = pcm16_bytes(0.05)
    assert moonshine_engine_instance.recognize_sync(audio_bytes, sample_rate=16000) == ""


def test_recognize_processes_valid_audio(moonshine_engine_instance):
    audio_bytes = pcm16_bytes(1.0)
    result = moonshine_engine_instance.recognize_sync(audio_bytes, sample_rate=16000)
    assert "test" in result.lower()


def _make_mock_transcriber() -> tuple[Mock, Mock]:
    """Build a Transcriber+Stream mock pair compatible with MoonshineStreamSession internals."""
    stream = Mock()
    stream._handle = 2
    stream._lib = Mock()
    stream._lib.moonshine_transcribe_add_audio_to_stream = Mock(return_value=0)
    stream._lib.moonshine_stop_stream = Mock(return_value=0)
    stream._stream_time = 0.0
    stream._last_update_time = 0.0
    stream._update_interval = 1.5
    stream.update_transcription = Mock()
    stream.start = Mock()
    stream.close = Mock()
    stream.add_listener = Mock()
    transcriber = Mock()
    transcriber._handle = 1
    transcriber.create_stream = Mock(return_value=stream)
    stream._transcriber = transcriber
    return transcriber, stream


class TestStreamSessionStop:
    """The native Moonshine stream silently drops audio added after the last
    ``transcription_interval`` boundary unless we force-flush before stopping it.
    These tests pin down that contract for the session wrapper."""

    def test_stop_drains_queued_audio_before_native_stop(self):
        from vocalance.app.services.dictation_flow.speech_recognition.moonshine_engine import MoonshineStreamSession

        transcriber, stream = _make_mock_transcriber()
        loop = asyncio.new_event_loop()
        try:
            sess = MoonshineStreamSession(
                transcriber=transcriber,
                loop=loop,
                on_partial=AsyncMock(),
                on_final=AsyncMock(),
                ms_config=MoonshineStreamingConfig(),
            )
            chunk = pcm16_bytes(0.03)
            for _ in range(10):
                sess.add_audio_pcm16(chunk, 16000)
            sess.stop()
        finally:
            loop.close()

        assert stream._lib.moonshine_transcribe_add_audio_to_stream.call_count == 10

    def test_stop_deactivates_vad_then_force_flushes(self):
        """VAD must be deactivated (C moonshine_stop_stream) before the final FORCE_UPDATE
        so the tail audio is captured without re-emitting already-finalized segments."""
        from vocalance.app.services.dictation_flow.speech_recognition.moonshine_engine import MoonshineStreamSession

        transcriber, stream = _make_mock_transcriber()
        call_log: list[str] = []
        stream._lib.moonshine_stop_stream.side_effect = lambda *_: call_log.append("native_stop")
        stream.update_transcription.side_effect = lambda flags=0: call_log.append(f"update({flags})")

        loop = asyncio.new_event_loop()
        try:
            sess = MoonshineStreamSession(
                transcriber=transcriber,
                loop=loop,
                on_partial=AsyncMock(),
                on_final=AsyncMock(),
                ms_config=MoonshineStreamingConfig(),
            )
            sess.add_audio_pcm16(pcm16_bytes(0.03), 16000)
            sess.stop()
        finally:
            loop.close()

        assert "native_stop" in call_log
        assert "update(1)" in call_log
        assert call_log.index("native_stop") < call_log.index("update(1)")

    def test_worker_thread_terminates_on_stop(self):
        from vocalance.app.services.dictation_flow.speech_recognition.moonshine_engine import MoonshineStreamSession

        transcriber, _ = _make_mock_transcriber()
        loop = asyncio.new_event_loop()
        try:
            sess = MoonshineStreamSession(
                transcriber=transcriber,
                loop=loop,
                on_partial=AsyncMock(),
                on_final=AsyncMock(),
                ms_config=MoonshineStreamingConfig(),
            )
            assert sess._worker.is_alive()
            sess.stop()
            assert not sess._worker.is_alive()
        finally:
            loop.close()

    def test_add_audio_after_stop_is_ignored(self):
        from vocalance.app.services.dictation_flow.speech_recognition.moonshine_engine import MoonshineStreamSession

        transcriber, stream = _make_mock_transcriber()
        loop = asyncio.new_event_loop()
        try:
            sess = MoonshineStreamSession(
                transcriber=transcriber,
                loop=loop,
                on_partial=AsyncMock(),
                on_final=AsyncMock(),
                ms_config=MoonshineStreamingConfig(),
            )
            sess.stop()
            calls_before = stream._lib.moonshine_transcribe_add_audio_to_stream.call_count
            sess.add_audio_pcm16(pcm16_bytes(0.03), 16000)
            assert stream._lib.moonshine_transcribe_add_audio_to_stream.call_count == calls_before
        finally:
            loop.close()
