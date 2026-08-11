from unittest.mock import Mock

import numpy as np
import pytest

from vocalance.app.events.core_events import CustomSoundRecognizedEvent
from vocalance.app.events.sound_events import SoundToCommandMappingUpdatedEvent, SoundTrainingInitiatedEvent


def test_preprocess_audio_chunk_normalizes(sound_service):
    audio_int16 = np.array([1000, -2000, 3000, -32768], dtype=np.int16)

    out = sound_service._preprocess_audio_chunk(audio_int16.tobytes())

    assert out.dtype == np.float32
    assert len(out) == 4
    assert np.max(np.abs(out)) <= 1.0
    assert np.isclose(out[0], 1000 / 32768.0)


@pytest.mark.parametrize("bad", [b"", "not-bytes"])
def test_preprocess_audio_chunk_rejects_invalid(sound_service, bad):
    with pytest.raises(ValueError):
        sound_service._preprocess_audio_chunk(bad)


@pytest.mark.asyncio
async def test_handle_audio_chunk_publishes_custom(sound_service, mock_recognizer, mock_event_bus):
    sound_service.is_initialized = True
    mock_recognizer.recognize_sound.return_value = ("click", 0.8)
    mock_recognizer.get_mapping.return_value = "copy"

    event = Mock()
    event.audio_chunk = np.array([1000, -2000], dtype=np.int16).tobytes()
    event.sample_rate = 16000

    await sound_service._handle_audio_chunk(event)

    published = mock_event_bus.publish.call_args[0][0]
    assert isinstance(published, CustomSoundRecognizedEvent)
    assert published.label == "click"
    assert published.confidence == 0.8
    assert published.mapped_command == "copy"


@pytest.mark.asyncio
@pytest.mark.parametrize("result", [None, ("esc50_breathing", 0.7)])
async def test_handle_audio_chunk_suppresses_non_custom(sound_service, mock_recognizer, mock_event_bus, result):
    sound_service.is_initialized = True
    mock_recognizer.recognize_sound.return_value = result

    event = Mock()
    event.audio_chunk = np.array([1000, -2000], dtype=np.int16).tobytes()
    event.sample_rate = 16000

    await sound_service._handle_audio_chunk(event)

    mock_event_bus.publish.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("requested,expected", [(0, 1), (5, 5), (10000, 1000)])
async def test_start_training_clamps_samples(sound_service, mock_event_bus, requested, expected):
    sound_service.is_initialized = True

    await sound_service.start_training_session("click", requested)

    published = mock_event_bus.publish.call_args[0][0]
    assert isinstance(published, SoundTrainingInitiatedEvent)
    assert published.total_samples == expected


@pytest.mark.asyncio
async def test_start_training_rejects_when_active(sound_service):
    sound_service.is_initialized = True
    sound_service._training_active = True

    result = await sound_service.start_training_session("click", 5)

    assert result is False


@pytest.mark.asyncio
async def test_map_sound_publishes_update_event(sound_service, mock_recognizer, mock_event_bus):
    mock_recognizer.set_mapping.return_value = True

    result = await sound_service.map_sound_to_command("click", "copy")

    assert result is True
    updates = [
        call[0][0] for call in mock_event_bus.publish.call_args_list if isinstance(call[0][0], SoundToCommandMappingUpdatedEvent)
    ]
    assert len(updates) == 1
    assert updates[0].sound_label == "click"
    assert updates[0].command_phrase == "copy"
    assert updates[0].success is True


@pytest.mark.asyncio
async def test_finish_training_without_active_returns_false(sound_service):
    result = await sound_service.finish_training()

    assert result is False


@pytest.mark.asyncio
async def test_finish_training_trains_and_resets(sound_service, mock_recognizer):
    mock_recognizer.train_sound.return_value = True
    sound_service._training_active = True
    sound_service._current_training_label = "click"
    sound_service._training_samples = [(np.zeros(10, dtype=np.float32), 16000)]

    result = await sound_service.finish_training()

    assert result is True
    mock_recognizer.train_sound.assert_called_once()
    assert sound_service.is_training_active() is False


def test_settings_changed_updates_confidence_threshold(sound_service, mock_recognizer):
    event = Mock()
    event.updated_settings = {"sound_recognizer.confidence_threshold": 0.4}

    sound_service._handle_settings_changed(event)

    mock_recognizer.on_confidence_threshold_updated.assert_called_once()
