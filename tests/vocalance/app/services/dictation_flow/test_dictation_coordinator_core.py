import asyncio
from unittest.mock import AsyncMock

import pytest
from conftest import skip_if_headless

skip_if_headless()

from vocalance.app.services.dictation_flow.dictation_coordinator import DictationSegmentPipeline, substitute_alias_placeholders
from vocalance.app.services.dictation_flow.types import DictationMode, DictationSession, DictationState


@pytest.mark.parametrize(
    "from_state,to_state",
    [
        (DictationState.IDLE, DictationState.PROCESSING_LLM),
        (DictationState.PROCESSING_LLM, DictationState.RECORDING),
        (DictationState.SHUTTING_DOWN, DictationState.IDLE),
        (DictationState.SHUTTING_DOWN, DictationState.RECORDING),
    ],
)
def test_set_state_rejects_invalid_transition(dictation_coordinator, from_state, to_state):
    dictation_coordinator.current_state = from_state
    with pytest.raises(ValueError):
        dictation_coordinator.set_state(to_state)


@pytest.mark.parametrize(
    "from_state,to_state",
    [
        (DictationState.IDLE, DictationState.RECORDING),
        (DictationState.IDLE, DictationState.SHUTTING_DOWN),
        (DictationState.RECORDING, DictationState.PROCESSING_LLM),
        (DictationState.RECORDING, DictationState.IDLE),
        (DictationState.PROCESSING_LLM, DictationState.IDLE),
    ],
)
def test_set_state_accepts_valid_transition(dictation_coordinator, from_state, to_state):
    dictation_coordinator.current_state = from_state
    dictation_coordinator.set_state(to_state)
    assert dictation_coordinator.current_state == to_state


@pytest.mark.parametrize(
    "text,alias_map,expected",
    [
        ("say vocalancealias0 now", {"vocalancealias0": "hello world"}, "say hello world now"),
        ("say VOCALANCEALIAS0 now", {"vocalancealias0": "hello world"}, "say hello world now"),
        ("a vocalancealias0 b vocalancealias1", {"vocalancealias0": "x", "vocalancealias1": "y"}, "a x b y"),
        ("nothing here", {}, "nothing here"),
    ],
)
def test_substitute_alias_placeholders(text, alias_map, expected):
    assert substitute_alias_placeholders(text, alias_map) == expected


@pytest.mark.parametrize("noise", ["?", ".", "  ", "！"])
def test_segment_pipeline_prepare_final_drops_noise(app_config, noop_alias_service, noise):
    pipeline = DictationSegmentPipeline(app_config.dictation, noop_alias_service)
    session = DictationSession(session_id="s", mode=DictationMode.HIDDEN, start_time=0.0)
    assert pipeline.prepare_final(noise, session) == ""


def test_segment_pipeline_prepare_final_processes_real_text(app_config, noop_alias_service):
    pipeline = DictationSegmentPipeline(app_config.dictation, noop_alias_service)
    session = DictationSession(session_id="s", mode=DictationMode.HIDDEN, start_time=0.0)
    assert pipeline.prepare_final("hello world", session) == "hello world"


def test_type_committed_text_is_held_until_stop(dictation_coordinator):
    coordinator = dictation_coordinator
    coordinator.alias_service.extract_aliases = lambda text: (text, {})
    coordinator.current_state = DictationState.RECORDING
    coordinator.current_session = DictationSession(
        session_id="s",
        mode=DictationMode.TYPE,
        start_time=0.0,
        active_modifiers={"strip", "diminish"},
    )
    coordinator.text_service.input_text = AsyncMock()

    asyncio.run(coordinator.asr.on_committed("Hello there"))

    coordinator.text_service.input_text.assert_not_called()
    assert coordinator.asr.streaming_finalized_text == "hello there"
    assert coordinator.current_session.accumulated_text == "hello there"


@pytest.mark.parametrize("mode", [DictationMode.TYPE, DictationMode.HIDDEN, DictationMode.VISUAL])
def test_stop_pastes_the_full_transcript_once(dictation_coordinator, mode):
    coordinator = dictation_coordinator
    coordinator.alias_service.extract_aliases = lambda text: (text, {})
    coordinator.alias_service.apply_substitutions = lambda text: text
    session = DictationSession(session_id="s", mode=mode, start_time=0.0, accumulated_text="hello")
    coordinator.current_state = DictationState.RECORDING
    coordinator.current_session = session
    coordinator.asr.streaming_finalized_text = "hello"
    coordinator.asr.halt_streaming_capturer = lambda: ("hello", "world")
    coordinator.text_service.input_text = AsyncMock()

    asyncio.run(coordinator.stop_streaming_mode(session))

    coordinator.text_service.input_text.assert_awaited_once()
    assert coordinator.text_service.input_text.await_args.kwargs["text"] == "hello world"
    assert coordinator.current_state == DictationState.IDLE
