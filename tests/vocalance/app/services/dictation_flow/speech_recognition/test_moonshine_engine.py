import numpy as np
import pytest


@pytest.mark.parametrize(
    "audio_bytes",
    [b"", np.zeros(int(0.05 * 16000), dtype=np.int16).tobytes()],
)
def test_recognize_returns_empty_for_empty_or_too_short_audio(moonshine_engine_instance, audio_bytes):
    assert moonshine_engine_instance.recognize_sync(audio_bytes, sample_rate=16000) == ""


def test_recognize_returns_normalized_transcript(moonshine_engine_instance):
    audio_bytes = np.zeros(16000, dtype=np.int16).tobytes()
    assert moonshine_engine_instance.recognize_sync(audio_bytes, sample_rate=16000) == "test recognition"


def test_stop_feeds_all_queued_audio_to_native_stream(moonshine_stream_session, mock_moonshine_transcriber):
    _, stream = mock_moonshine_transcriber
    chunk = np.zeros(int(0.03 * 16000), dtype=np.int16).tobytes()
    for _ in range(10):
        moonshine_stream_session.add_audio_pcm16(chunk, 16000)
    moonshine_stream_session.stop()
    assert stream._lib.moonshine_transcribe_add_audio_to_stream.call_count == 10


def test_stop_deactivates_vad_before_force_flush(moonshine_stream_session, mock_moonshine_transcriber):
    """VAD must be deactivated before the final FORCE_UPDATE flush so the tail audio is
    captured without re-emitting already-finalized segments."""
    _, stream = mock_moonshine_transcriber
    call_log: list[str] = []
    stream._lib.moonshine_stop_stream.side_effect = lambda *_: call_log.append("native_stop")
    stream.update_transcription.side_effect = lambda flags=0: call_log.append(f"update({flags})")

    moonshine_stream_session.add_audio_pcm16(np.zeros(int(0.03 * 16000), dtype=np.int16).tobytes(), 16000)
    moonshine_stream_session.stop()

    assert "native_stop" in call_log
    assert "update(1)" in call_log
    assert call_log.index("native_stop") < call_log.index("update(1)")


def test_worker_thread_terminates_after_stop(moonshine_stream_session):
    assert moonshine_stream_session._worker.is_alive()
    moonshine_stream_session.stop()
    assert not moonshine_stream_session._worker.is_alive()


def test_add_audio_after_stop_is_dropped(moonshine_stream_session, mock_moonshine_transcriber):
    _, stream = mock_moonshine_transcriber
    moonshine_stream_session.stop()
    calls_before = stream._lib.moonshine_transcribe_add_audio_to_stream.call_count
    moonshine_stream_session.add_audio_pcm16(np.zeros(int(0.03 * 16000), dtype=np.int16).tobytes(), 16000)
    assert stream._lib.moonshine_transcribe_add_audio_to_stream.call_count == calls_before
