"""Tests for dictation coordinator helpers used with voice modifiers (explicit expectations)."""

from typing import Optional
from unittest.mock import AsyncMock, Mock, patch

import pytest

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.events.dictation_events import DictationModifierId
from vocalance.app.services.audio.dictation_handling.dictation_coordinator import (
    DictationCoordinator,
    DictationMode,
)


@pytest.fixture
def coordinator_minimal() -> DictationCoordinator:
    """Coordinator with real config; heavy services mocked."""
    bus = Mock()
    bus.subscribe = Mock()
    bus.publish = AsyncMock()
    storage = Mock()
    with patch(
        "vocalance.app.services.audio.dictation_handling.dictation_coordinator.TextInputService"
    ), patch("vocalance.app.services.audio.dictation_handling.dictation_coordinator.LLMService"), patch(
        "vocalance.app.services.audio.dictation_handling.dictation_coordinator.AgenticPromptService"
    ), patch("vocalance.app.services.audio.dictation_handling.dictation_coordinator.DictationAliasService"):
        coord = DictationCoordinator(
            event_bus=bus,
            config=GlobalAppConfig(),
            storage=storage,
        )
    coord.alias_service = Mock()
    coord.alias_service.apply_substitutions = lambda t: t
    return coord


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
    add, skip = DictationCoordinator._dictation_segment_input_options(mode, modifier)
    assert add is expected_add_trailing
    assert skip is expected_skip_join


@pytest.mark.parametrize(
    "cleaned,expected_echo",
    [
        ("", True),
        ("hello world", False),
        ("offer", False),
        ("spell", True),
        ("upper", True),
        ("spelling", True),
    ],
)
def test_is_likely_modifier_asr_echo(coordinator_minimal: DictationCoordinator, cleaned: str, expected_echo: bool) -> None:
    assert coordinator_minimal._is_likely_modifier_asr_echo(cleaned) is expected_echo


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("hello upper world", "hello world"),
        ("test Upper case phrase", "test case phrase"),
        ("no modifiers here", "no modifiers here"),
    ],
)
def test_clean_text_strips_default_modifier_phrases(
    coordinator_minimal: DictationCoordinator, raw: str, expected: str
) -> None:
    assert coordinator_minimal._clean_text(raw) == expected


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
    assert DictationCoordinator._is_isolated_stt_noise_fragment(text) is expected_noise


def test_build_modifier_phrases_lc_matches_config(coordinator_minimal: DictationCoordinator) -> None:
    cfg = coordinator_minimal.config.dictation
    assert coordinator_minimal._modifier_phrases_lc == (
        cfg.modifier_upper_phrase.strip().lower(),
        cfg.modifier_capitals_phrase.strip().lower(),
        cfg.modifier_camel_phrase.strip().lower(),
        cfg.modifier_snake_phrase.strip().lower(),
        cfg.modifier_spelling_phrase.strip().lower(),
    )
