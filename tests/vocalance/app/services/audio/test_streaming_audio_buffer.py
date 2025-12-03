import asyncio

import numpy as np
import pytest

from vocalance.app.services.audio.streaming_audio_buffer import StreamingAudioBuffer


@pytest.fixture
def buffer():
    return StreamingAudioBuffer(sample_rate=16000)


@pytest.fixture
def audio_chunk_int16():
    return np.random.randint(-1000, 1000, size=800, dtype=np.int16)


@pytest.fixture
def audio_chunk_float32():
    return np.random.uniform(-0.5, 0.5, size=800).astype(np.float32)


@pytest.mark.asyncio
async def test_add_chunk_initializes_buffer(buffer, audio_chunk_int16):
    await buffer.add_chunk(audio_chunk_int16)

    duration = await buffer.get_duration()
    assert duration > 0.0  # 800 samples at 16000 Hz = 0.05 seconds


@pytest.mark.asyncio
async def test_buffer_maintains_max_capacity(buffer):
    """Test that buffer trims when exceeding maximum capacity."""
    chunk = np.zeros(16000, dtype=np.int16)

    for _ in range(50):
        await buffer.add_chunk(chunk)

    duration = await buffer.get_duration()
    assert duration <= 45.0  # Buffer should not exceed 45 seconds


@pytest.mark.asyncio
async def test_get_audio_for_transcription_returns_none_when_empty(buffer):
    result = await buffer.get_audio_for_transcription()
    assert result is None


@pytest.mark.asyncio
async def test_get_audio_for_transcription_returns_audio_and_duration(buffer, audio_chunk_int16):
    await buffer.add_chunk(audio_chunk_int16)

    result = await buffer.get_audio_for_transcription()

    assert result is not None
    audio_bytes, duration = result
    assert isinstance(audio_bytes, bytes)
    assert isinstance(duration, float)
    assert duration > 0.0


@pytest.mark.asyncio
async def test_timestamp_offset_tracking(buffer):
    """Test that timestamp offset advances correctly."""
    initial_offset = await buffer.get_timestamp_offset()
    assert initial_offset == 0.0

    await buffer.advance_timestamp_offset(1.5)
    new_offset = await buffer.get_timestamp_offset()
    assert new_offset == 1.5


@pytest.mark.asyncio
async def test_untranscribed_duration_calculation(buffer, audio_chunk_int16):
    """Test calculation of unprocessed audio duration."""
    # Empty buffer
    duration = await buffer.get_unprocessed_duration()
    assert duration == 0.0

    # With audio
    await buffer.add_chunk(audio_chunk_int16)
    duration = await buffer.get_unprocessed_duration()
    assert duration > 0.0

    # After advancing timestamp offset
    await buffer.advance_timestamp_offset(0.025)
    new_duration = await buffer.get_unprocessed_duration()
    assert new_duration < duration


@pytest.mark.asyncio
async def test_clear_resets_state(buffer, audio_chunk_int16):
    """Test that clear resets all buffer state."""
    await buffer.add_chunk(audio_chunk_int16)
    await buffer.advance_timestamp_offset(1.0)

    await buffer.clear()

    # Verify state is reset
    duration = await buffer.get_unprocessed_duration()
    assert duration == 0.0

    offset = await buffer.get_timestamp_offset()
    assert offset == 0.0


@pytest.mark.asyncio
async def test_transcription_with_timestamp_offset(buffer):
    """Test that transcription respects timestamp offset."""
    chunk1 = np.ones(16000, dtype=np.int16) * 100
    chunk2 = np.ones(16000, dtype=np.int16) * 200

    await buffer.add_chunk(chunk1)
    await buffer.add_chunk(chunk2)
    await buffer.advance_timestamp_offset(1.0)

    audio_bytes, duration = await buffer.get_audio_for_transcription()

    # Should return only unprocessed portion
    assert duration < 2.0


@pytest.mark.asyncio
async def test_concurrent_operations_safe(buffer):
    """Test that concurrent add/read operations are thread-safe."""
    chunks = [np.random.randint(-1000, 1000, size=800, dtype=np.int16) for _ in range(10)]

    # Add chunks concurrently
    tasks = [buffer.add_chunk(chunk) for chunk in chunks]
    await asyncio.gather(*tasks)

    # Verify buffer has content
    duration = await buffer.get_duration()
    assert duration > 0.0

    # Concurrent reads should work
    await buffer.add_chunk(chunks[0])
    tasks = [buffer.get_audio_for_transcription(), buffer.get_unprocessed_duration(), buffer.get_timestamp_offset()]
    results = await asyncio.gather(*tasks)
    assert len(results) == 3


@pytest.mark.asyncio
async def test_timestamp_beyond_buffer_returns_none(buffer, audio_chunk_int16):
    """Test that requesting audio beyond buffer returns None."""
    await buffer.add_chunk(audio_chunk_int16)
    await buffer.advance_timestamp_offset(10.0)  # Advance far beyond buffer content

    result = await buffer.get_audio_for_transcription()
    assert result is None
