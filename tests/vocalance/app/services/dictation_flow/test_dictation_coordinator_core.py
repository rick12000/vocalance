import pytest

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
    session = DictationSession(session_id="s", mode=DictationMode.STANDARD, start_time=0.0)
    assert pipeline.prepare_final(noise, session) == ""


def test_segment_pipeline_prepare_final_processes_real_text(app_config, noop_alias_service):
    pipeline = DictationSegmentPipeline(app_config.dictation, noop_alias_service)
    session = DictationSession(session_id="s", mode=DictationMode.STANDARD, start_time=0.0)
    assert pipeline.prepare_final("hello world", session) == "hello world"
