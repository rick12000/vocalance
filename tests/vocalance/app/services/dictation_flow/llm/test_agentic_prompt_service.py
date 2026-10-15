from datetime import datetime

import pytest

from vocalance.app.events.dictation_events import AgenticPromptUiOperationEvent
from vocalance.app.services.dictation_flow.llm.agentic_prompt_service import (
    AgenticPromptService,
    _CLARITY_FORMAT_PROMPT,
    _DEFAULT_PROMPT,
    _DEFAULT_PROMPT_NAME,
    _NON_NEGOTIABLE_CONSTRAINTS,
    _SYSTEM_PROMPTS,
)
from vocalance.app.services.dictation_flow.llm.dictation_messages import (
    _DICTATION_SYSTEM_STUB,
    _TRANSCRIPT_BEGIN,
    _TRANSCRIPT_END,
)
from vocalance.app.services.dictation_flow.llm.formatter_output_rules import OUTPUT_SLOP_AND_FORMATTING_RULE
from vocalance.app.services.dictation_flow.llm.llm_service import _AMEND_SYSTEM_BASE
from vocalance.app.services.storage.storage_models import AgenticPrompt, AgenticPromptsData


def test_shipped_prompts_share_non_negotiable_constraints():
    shipped = [_DEFAULT_PROMPT, *(spec.text for spec in _SYSTEM_PROMPTS)]
    for prompt_text in shipped:
        assert _NON_NEGOTIABLE_CONSTRAINTS in prompt_text
        assert "Your only task" in prompt_text


def test_clarity_allows_reorder_default_does_not():
    assert "reorder and rephrase for clarity" in _CLARITY_FORMAT_PROMPT
    assert "Do not summarize, change style, or reorder ideas" in _DEFAULT_PROMPT


def test_default_and_clarity_share_numbers_and_differ_on_fidelity():
    from vocalance.app.services.dictation_flow.llm.agentic_prompt_service import _NUMBERS_AND_DATES_BRIEF

    assert _NUMBERS_AND_DATES_BRIEF.strip() in _DEFAULT_PROMPT
    assert _NUMBERS_AND_DATES_BRIEF.strip() in _CLARITY_FORMAT_PROMPT
    assert "High-fidelity" in _DEFAULT_PROMPT
    assert "Actively structure" in _CLARITY_FORMAT_PROMPT


def test_shipped_prompts_avoid_semicolons_and_em_dashes():
    shipped = [_DEFAULT_PROMPT, *(spec.text for spec in _SYSTEM_PROMPTS)]
    for prompt_text in shipped:
        assert ";" not in prompt_text
        assert "—" not in prompt_text
        assert "–" not in prompt_text


async def test_initialize_creates_single_default(agentic_prompt_service):
    defaults = [p for p in agentic_prompt_service.get_all_prompts() if p.is_default]
    assert len(defaults) == 1
    assert agentic_prompt_service.current_prompt_id == defaults[0].id


async def test_initialize_creates_system_prompts(agentic_prompt_service):
    system_prompts = [p for p in agentic_prompt_service.get_all_prompts() if p.system_key is not None]
    assert len(system_prompts) == 2
    keys = {p.system_key for p in system_prompts}
    assert keys == {"email_format", "clarity_format"}


async def test_initialize_syncs_stale_default_name(mock_event_bus, app_config, agentic_prompts_storage):
    existing = AgenticPrompt(
        id="d1", text="x", name="Old Default", created_at=datetime.now().isoformat(), is_default=True
    )
    agentic_prompts_storage.read.return_value = AgenticPromptsData(prompts=[existing], current_prompt_id="d1")

    service = AgenticPromptService(event_bus=mock_event_bus, config=app_config, storage=agentic_prompts_storage)
    await service.initialize()

    defaults = [p for p in service.get_all_prompts() if p.is_default]
    assert defaults[0].name == _DEFAULT_PROMPT_NAME


async def test_ensure_system_prompts_syncs_stale_name(mock_event_bus, app_config, agentic_prompts_storage):
    default = AgenticPrompt(
        id="d1", text="x", name=_DEFAULT_PROMPT_NAME, created_at=datetime.now().isoformat(), is_default=True
    )
    stale_email = AgenticPrompt(
        id="e1",
        text="old",
        name="Old Email Name",
        created_at=datetime.now().isoformat(),
        system_key="email_format",
        system_canonical_name="Old Email Name",
    )
    agentic_prompts_storage.read.return_value = AgenticPromptsData(
        prompts=[default, stale_email], current_prompt_id="d1"
    )

    service = AgenticPromptService(event_bus=mock_event_bus, config=app_config, storage=agentic_prompts_storage)
    await service.initialize()

    email = next(p for p in service.get_all_prompts() if p.system_key == "email_format")
    assert email.name == "Email Formatter"


async def test_ensure_system_prompts_preserves_user_customised_name(mock_event_bus, app_config, agentic_prompts_storage):
    default = AgenticPrompt(
        id="d1", text="x", name=_DEFAULT_PROMPT_NAME, created_at=datetime.now().isoformat(), is_default=True
    )
    custom_email = AgenticPrompt(
        id="e1",
        text="custom",
        name="My Email Style",
        created_at=datetime.now().isoformat(),
        system_key="email_format",
        system_canonical_name="Email Formatter",
    )
    agentic_prompts_storage.read.return_value = AgenticPromptsData(
        prompts=[default, custom_email], current_prompt_id="d1"
    )

    service = AgenticPromptService(event_bus=mock_event_bus, config=app_config, storage=agentic_prompts_storage)
    await service.initialize()

    email = next(p for p in service.get_all_prompts() if p.system_key == "email_format")
    assert email.name == "My Email Style"


async def test_initialize_does_not_duplicate_system_prompts(mock_event_bus, app_config, agentic_prompts_storage):
    default = AgenticPrompt(
        id="d1", text="x", name=_DEFAULT_PROMPT_NAME, created_at=datetime.now().isoformat(), is_default=True
    )
    email = AgenticPrompt(
        id="e1",
        text="e",
        name="Email Formatter",
        created_at=datetime.now().isoformat(),
        system_key="email_format",
        system_canonical_name="Email Formatter",
        system_canonical_text="e",
    )
    agentic_prompts_storage.read.return_value = AgenticPromptsData(
        prompts=[default, email], current_prompt_id="d1"
    )

    service = AgenticPromptService(event_bus=mock_event_bus, config=app_config, storage=agentic_prompts_storage)
    await service.initialize()
    await service.initialize()

    system_prompts = [p for p in service.get_all_prompts() if p.system_key is not None]
    assert len(system_prompts) == 2


async def test_add_prompt_trims_and_stores(agentic_prompt_service):
    before = len(agentic_prompt_service.get_all_prompts())

    pid = await agentic_prompt_service.add_prompt("  hello  ", "  Greeting  ")

    prompts = agentic_prompt_service.get_all_prompts()
    assert len(prompts) == before + 1
    added = next(p for p in prompts if p.id == pid)
    assert added.text == "hello"
    assert added.name == "Greeting"


async def test_delete_current_prompt_reassigns_selection(agentic_prompt_service):
    pid = await agentic_prompt_service.add_prompt("text", "name")
    agentic_prompt_service.set_current_prompt(pid)

    result = await agentic_prompt_service.delete_prompt(pid)

    assert result is True
    assert pid not in [p.id for p in agentic_prompt_service.get_all_prompts()]
    assert agentic_prompt_service.current_prompt_id != pid
    assert agentic_prompt_service.current_prompt_id in agentic_prompt_service.prompts


async def test_delete_rejects_default_and_unknown(agentic_prompt_service):
    default = agentic_prompt_service.get_default_prompt()
    before = len(agentic_prompt_service.get_all_prompts())

    assert await agentic_prompt_service.delete_prompt(default.id) is False
    assert await agentic_prompt_service.delete_prompt("missing") is False
    assert len(agentic_prompt_service.get_all_prompts()) == before


async def test_delete_rejects_system_prompt(agentic_prompt_service):
    system_prompt = next(p for p in agentic_prompt_service.get_all_prompts() if p.system_key is not None)
    before = len(agentic_prompt_service.get_all_prompts())

    assert await agentic_prompt_service.delete_prompt(system_prompt.id) is False
    assert len(agentic_prompt_service.get_all_prompts()) == before


async def test_edit_prompt_trims_and_updates(agentic_prompt_service):
    pid = await agentic_prompt_service.add_prompt("old", "Old")

    result = await agentic_prompt_service.edit_prompt(pid, "  New  ", "  New text  ")

    assert result is True
    edited = next(p for p in agentic_prompt_service.get_all_prompts() if p.id == pid)
    assert edited.name == "New"
    assert edited.text == "New text"


async def test_edit_rejects_default_and_unknown(agentic_prompt_service):
    default = agentic_prompt_service.get_default_prompt()

    assert await agentic_prompt_service.edit_prompt(default.id, "x", "y") is False
    assert await agentic_prompt_service.edit_prompt("missing", "x", "y") is False
    assert agentic_prompt_service.get_default_prompt().text != "y"


async def test_edit_allows_system_prompt(agentic_prompt_service):
    system_prompt = next(p for p in agentic_prompt_service.get_all_prompts() if p.system_key is not None)

    result = await agentic_prompt_service.edit_prompt(system_prompt.id, "New Name", "New text")

    assert result is True
    updated = next(p for p in agentic_prompt_service.get_all_prompts() if p.id == system_prompt.id)
    assert updated.name == "New Name"
    assert updated.text == "New text"


async def test_set_current_selects_existing_and_rejects_unknown(agentic_prompt_service):
    pid = await agentic_prompt_service.add_prompt("body", "Name")

    assert agentic_prompt_service.set_current_prompt(pid) is True
    assert agentic_prompt_service.get_current_prompt() == "body"
    assert agentic_prompt_service.set_current_prompt("missing") is False
    assert agentic_prompt_service.current_prompt_id == pid


async def test_ui_operation_routes_add_select_delete(agentic_prompt_service):
    await agentic_prompt_service.handle_agentic_prompt_ui_operation(
        AgenticPromptUiOperationEvent(op="add", prompt_text="hi", name="Custom")
    )
    custom = next(p for p in agentic_prompt_service.get_all_prompts() if p.name == "Custom")

    await agentic_prompt_service.handle_agentic_prompt_ui_operation(
        AgenticPromptUiOperationEvent(op="select", prompt_id=custom.id)
    )
    assert agentic_prompt_service.current_prompt_id == custom.id

    await agentic_prompt_service.handle_agentic_prompt_ui_operation(
        AgenticPromptUiOperationEvent(op="delete", prompt_id=custom.id)
    )
    assert custom.id not in [p.id for p in agentic_prompt_service.get_all_prompts()]


@pytest.mark.parametrize("extra", ["", "Formatter rules here"])
def test_build_messages_puts_rules_and_transcript_in_user(llm_service, extra):
    messages = llm_service.build_messages(extra, "raw input")

    assert len(messages) == 2
    assert _DICTATION_SYSTEM_STUB in messages[0]["content"]
    assert OUTPUT_SLOP_AND_FORMATTING_RULE in messages[0]["content"]
    user = messages[1]["content"]
    assert _TRANSCRIPT_BEGIN in user
    assert _TRANSCRIPT_END in user
    assert "raw input" in user
    if extra:
        assert extra in user
    else:
        assert _DEFAULT_PROMPT in user


async def test_ensure_default_prompt_syncs_text_from_code(mock_event_bus, app_config, agentic_prompts_storage):
    stale = AgenticPrompt(
        id="d1",
        text="old bundled default prompt",
        name=_DEFAULT_PROMPT_NAME,
        created_at=datetime.now().isoformat(),
        is_default=True,
    )
    agentic_prompts_storage.read.return_value = AgenticPromptsData(prompts=[stale], current_prompt_id="d1")

    service = AgenticPromptService(event_bus=mock_event_bus, config=app_config, storage=agentic_prompts_storage)
    await service.initialize()

    assert service.get_default_prompt().text == service.default_prompt_text


@pytest.mark.parametrize("extra", ["", "Keep it terse"])
def test_build_amend_messages_embeds_base_extra_and_inputs(llm_service, extra):
    messages = llm_service.build_amend_messages(extra, "CLIP CONTENT", "SPOKEN PROMPT")

    assert len(messages) == 2
    system = messages[0]["content"]
    assert system.startswith(_AMEND_SYSTEM_BASE)
    assert OUTPUT_SLOP_AND_FORMATTING_RULE in system
    assert (extra in system) if extra else True
    user = messages[1]["content"]
    assert "CLIP CONTENT" in user
    assert "SPOKEN PROMPT" in user
