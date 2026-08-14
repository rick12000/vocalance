from datetime import datetime

import pytest

from vocalance.app.events.dictation_events import AgenticPromptUiOperationEvent
from vocalance.app.services.dictation_flow.llm.agentic_prompt_service import (
    AgenticPromptService,
    _DEFAULT_PROMPT_NAME,
    _SYSTEM_PROMPTS,
)
from vocalance.app.services.dictation_flow.llm.llm_service import _AMEND_SYSTEM_BASE
from vocalance.app.services.storage.storage_models import AgenticPrompt, AgenticPromptsData


async def test_initialize_creates_single_default(agentic_prompt_service):
    defaults = [p for p in agentic_prompt_service.get_all_prompts() if p.is_default]
    assert len(defaults) == 1
    assert agentic_prompt_service.current_prompt_id == defaults[0].id


async def test_initialize_creates_system_prompts(agentic_prompt_service):
    system_prompts = [p for p in agentic_prompt_service.get_all_prompts() if p.system_key is not None]
    system_keys = {p.system_key for p in system_prompts}
    expected_keys = {key for key, _, _ in _SYSTEM_PROMPTS}
    assert system_keys == expected_keys
    assert all(not p.is_default for p in system_prompts)


async def test_initialize_syncs_stale_default_name(mock_event_bus, app_config, agentic_prompts_storage):
    existing = AgenticPrompt(id="d1", text="x", name="Default", created_at=datetime.now().isoformat(), is_default=True)
    agentic_prompts_storage.read.return_value = AgenticPromptsData(prompts=[existing], current_prompt_id="d1")

    service = AgenticPromptService(event_bus=mock_event_bus, config=app_config, storage=agentic_prompts_storage)
    await service.initialize()

    defaults = [p for p in service.get_all_prompts() if p.is_default]
    assert len(defaults) == 1
    assert defaults[0].name == _DEFAULT_PROMPT_NAME
    assert service.current_prompt_id == "d1"


async def test_ensure_system_prompts_syncs_stale_name(mock_event_bus, app_config, agentic_prompts_storage):
    existing_default = AgenticPrompt(
        id="d1", text="x", name=_DEFAULT_PROMPT_NAME, created_at=datetime.now().isoformat(), is_default=True
    )
    stale_system = AgenticPrompt(
        id="s1",
        text="email text",
        name="Email Format",
        created_at=datetime.now().isoformat(),
        system_key="email_format",
        system_canonical_name=None,
    )
    agentic_prompts_storage.read.return_value = AgenticPromptsData(
        prompts=[existing_default, stale_system], current_prompt_id="d1"
    )

    service = AgenticPromptService(event_bus=mock_event_bus, config=app_config, storage=agentic_prompts_storage)
    await service.initialize()

    synced = next(p for p in service.get_all_prompts() if p.system_key == "email_format")
    assert synced.name == "Email Formatter"
    assert synced.system_canonical_name == "Email Formatter"


async def test_ensure_system_prompts_preserves_user_customised_name(mock_event_bus, app_config, agentic_prompts_storage):
    existing_default = AgenticPrompt(
        id="d1", text="x", name=_DEFAULT_PROMPT_NAME, created_at=datetime.now().isoformat(), is_default=True
    )
    user_renamed = AgenticPrompt(
        id="s1",
        text="email text",
        name="My Email",
        created_at=datetime.now().isoformat(),
        system_key="email_format",
        system_canonical_name="Email Formatter",
    )
    agentic_prompts_storage.read.return_value = AgenticPromptsData(
        prompts=[existing_default, user_renamed], current_prompt_id="d1"
    )

    service = AgenticPromptService(event_bus=mock_event_bus, config=app_config, storage=agentic_prompts_storage)
    await service.initialize()

    preserved = next(p for p in service.get_all_prompts() if p.system_key == "email_format")
    assert preserved.name == "My Email"


async def test_initialize_does_not_duplicate_system_prompts(mock_event_bus, app_config, agentic_prompts_storage):
    existing_default = AgenticPrompt(
        id="d1", text="x", name="Default", created_at=datetime.now().isoformat(), is_default=True
    )
    existing_system = AgenticPrompt(
        id="s1",
        text="email text",
        name="Email Format",
        created_at=datetime.now().isoformat(),
        system_key="email_format",
    )
    agentic_prompts_storage.read.return_value = AgenticPromptsData(
        prompts=[existing_default, existing_system], current_prompt_id="d1"
    )

    service = AgenticPromptService(event_bus=mock_event_bus, config=app_config, storage=agentic_prompts_storage)
    await service.initialize()

    email_prompts = [p for p in service.get_all_prompts() if p.system_key == "email_format"]
    assert len(email_prompts) == 1


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
    original_key = system_prompt.system_key

    result = await agentic_prompt_service.edit_prompt(system_prompt.id, "New Name", "New text")

    assert result is True
    updated = next(p for p in agentic_prompt_service.get_all_prompts() if p.id == system_prompt.id)
    assert updated.name == "New Name"
    assert updated.text == "New text"
    assert updated.system_key == original_key


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


def test_build_messages_maps_roles(llm_service):
    messages = llm_service.build_messages("system instructions", "raw input")

    assert len(messages) == 2
    assert messages[0] == {"role": "system", "content": "system instructions"}
    assert messages[1] == {"role": "user", "content": "raw input"}


@pytest.mark.parametrize("extra", ["", "Keep it terse"])
def test_build_amend_messages_embeds_base_extra_and_inputs(llm_service, extra):
    messages = llm_service.build_amend_messages(extra, "CLIP CONTENT", "SPOKEN PROMPT")

    assert len(messages) == 2
    system = messages[0]["content"]
    assert system.startswith(_AMEND_SYSTEM_BASE)
    assert (extra in system) if extra else (system == _AMEND_SYSTEM_BASE)
    user = messages[1]["content"]
    assert "CLIP CONTENT" in user
    assert "SPOKEN PROMPT" in user
