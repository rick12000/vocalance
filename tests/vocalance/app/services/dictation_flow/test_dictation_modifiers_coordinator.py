from typing import Optional

import pytest

from vocalance.app.events.dictation_events import DictationModifierId
from vocalance.app.services.dictation_flow.postprocess.coordinator_segment_filters import (
    dictation_segment_input_options,
    is_isolated_stt_noise_fragment,
    is_likely_hallucination_fragment,
    remove_stop_trigger_word,
)
from vocalance.app.services.dictation_flow.postprocess.trigger_strip import (
    strip_config_phrases_case_insensitive,
    strip_dictation_triggers,
)
from vocalance.app.services.dictation_flow.types import DictationMode


@pytest.mark.parametrize(
    "mode,modifier,expected_add_trailing,expected_skip_join",
    [
        (DictationMode.STANDARD, None, True, False),
        (DictationMode.STANDARD, "upper", True, False),
        (DictationMode.STANDARD, "camel", False, True),
        (DictationMode.STANDARD, "snake", False, True),
        (DictationMode.STANDARD, "spelling", False, True),
        (DictationMode.TYPE, None, False, False),
        (DictationMode.TYPE, "camel", False, True),
    ],
)
def test_dictation_segment_input_options(
    mode: DictationMode,
    modifier: Optional[DictationModifierId],
    expected_add_trailing: bool,
    expected_skip_join: bool,
) -> None:
    modifiers = {modifier} if modifier else set()
    add_trailing, skip_join = dictation_segment_input_options(mode, modifiers)
    assert add_trailing is expected_add_trailing
    assert skip_join is expected_skip_join


@pytest.mark.parametrize(
    "text,expected_noise",
    [
        ("", True),
        ("?", True),
        (".", True),
        ("ab", False),
        ("hello", False),
    ],
)
def test_is_isolated_stt_noise_fragment(text: str, expected_noise: bool) -> None:
    assert is_isolated_stt_noise_fragment(text) is expected_noise


@pytest.mark.parametrize(
    "text,expected",
    [
        ("a normal sentence that should pass through untouched", False),
        ("ab ab ab ab ab ab ab ab ab ab ab ab", True),
        ("x", False),
    ],
)
def test_is_likely_hallucination_fragment(text: str, expected: bool) -> None:
    assert is_likely_hallucination_fragment(text) is expected


@pytest.mark.parametrize(
    "text,stop_trigger,expected",
    [
        ("stop the recording amber now", "amber", "stop the recording now"),
        ("shambers of commerce", "amber", "shambers of commerce"),
        ("nothing to remove", "amber", "nothing to remove"),
    ],
)
def test_remove_stop_trigger_word(text: str, stop_trigger: str, expected: str) -> None:
    assert remove_stop_trigger_word(text, stop_trigger) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("hello green fields", "hello fields"),
        ("now smart green tomorrow", "now tomorrow"),
        ("the amber lamp", "the lamp"),
        ("shambers of commerce", "shambers of commerce"),
        ("hello upper world", "hello world"),
        ("use spelling mode", "use mode"),
        ("spell check this", "spell check this"),
        ("no modifiers here", "no modifiers here"),
    ],
)
def test_strip_dictation_triggers(app_config, text: str, expected: str) -> None:
    assert strip_dictation_triggers(text, app_config.dictation) == expected


def test_strip_config_phrases_is_identity_on_empty_phrases() -> None:
    assert strip_config_phrases_case_insensitive("a b c", ()) == "a b c"
