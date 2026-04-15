import asyncio
from collections.abc import Iterator
from typing import Optional
from unittest.mock import AsyncMock, Mock, patch

import pytest

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.events.dictation_events import DictationModifierId
from vocalance.app.services.audio.dictation_handling.dictation_coordinator import DictationCoordinator
from vocalance.app.services.audio.dictation_handling.types import DictationMode
from vocalance.app.services.audio.dictation_handling.utils.coordinator_segment_filters import (
    dictation_segment_input_options,
    is_isolated_stt_noise_fragment,
)
from vocalance.app.services.audio.dictation_handling.utils.trigger_strip import strip_config_phrases_case_insensitive


@pytest.fixture
def coordinator_minimal() -> Iterator[DictationCoordinator]:
    """Coordinator with real config; heavy services mocked."""
    loop = asyncio.new_event_loop()
    bus = Mock()
    bus.subscribe = Mock()
    bus.publish = AsyncMock()
    storage = Mock()
    with patch("vocalance.app.services.audio.dictation_handling.dictation_coordinator.DictationTextInput"), patch(
        "vocalance.app.services.audio.dictation_handling.dictation_coordinator.LLMService"
    ), patch("vocalance.app.services.audio.dictation_handling.dictation_coordinator.AgenticPromptService"), patch(
        "vocalance.app.services.audio.dictation_handling.dictation_coordinator.DictationAliasService"
    ):
        coord = DictationCoordinator(
            event_bus=bus,
            config=GlobalAppConfig(),
            storage=storage,
            gui_event_loop=loop,
            stt_service=Mock(),
        )
    coord.alias_service = Mock()
    coord.alias_service.apply_substitutions = lambda t: t
    try:
        yield coord
    finally:
        loop.close()


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
    modifiers_set = {modifier} if modifier else set()
    add, skip = dictation_segment_input_options(mode, modifiers_set)
    assert add is expected_add_trailing
    assert skip is expected_skip_join


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("hello upper world", "hello world"),
        ("test Upper case phrase", "test case phrase"),
        ("no modifiers here", "no modifiers here"),
    ],
)
def test_clean_text_strips_default_modifier_phrases(coordinator_minimal: DictationCoordinator, raw: str, expected: str) -> None:
    assert coordinator_minimal.clean_text(raw) == expected


def test_clean_text_strips_stop_trigger_as_whole_word_only(coordinator_minimal: DictationCoordinator) -> None:
    """Stop trigger defaults to ``amber`` (DictationConfig), not the substring of unrelated words."""
    stop = coordinator_minimal.config.dictation.stop_trigger
    assert stop == "amber"
    assert coordinator_minimal.clean_text("the amber lamp") == "the lamp"
    assert coordinator_minimal.clean_text("shambers of commerce") == "shambers of commerce"


def test_clean_text_strips_start_and_multiword_smart_triggers(coordinator_minimal: DictationCoordinator) -> None:
    assert coordinator_minimal.clean_text("hello green fields") == "hello fields"
    assert coordinator_minimal.clean_text("now smart green tomorrow") == "now tomorrow"


def test_clean_text_keeps_spell_when_not_exact_modifier_phrase(coordinator_minimal: DictationCoordinator) -> None:
    """Only exact configured modifier phrases are removed; ``spell`` is not ``spelling``."""
    assert coordinator_minimal.clean_text("spell check this") == "spell check this"
    assert coordinator_minimal.clean_text("use spelling mode") == "use mode"


@pytest.mark.parametrize(
    "text,expected_noise",
    [
        ("", True),
        ("?", True),
        ("ab", False),
        (".", True),
        ("hello", False),
    ],
)
def test_is_isolated_stt_noise_fragment(text: str, expected_noise: bool) -> None:
    assert is_isolated_stt_noise_fragment(text) is expected_noise


def test_strip_config_phrases_idempotent_on_empty_phrases() -> None:
    assert strip_config_phrases_case_insensitive("a b c", ()) == "a b c"
