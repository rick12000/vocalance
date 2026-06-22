import asyncio

import pytest

from vocalance.app.events.command_management_events import CommandMappingsUpdatedEvent
from vocalance.app.services.command_flow.management.protected_terms_validator import ProtectedTermsValidator
from vocalance.app.services.storage.storage_models import MarksData, SoundMappingsData


@pytest.mark.asyncio
async def test_protected_terms_aggregate_registry_config_and_storage(
    protected_terms_validator, protected_terms_storage, app_config
):
    protected_terms_storage.read.side_effect = lambda model_type: (
        MarksData(marks={"My Mark": {"x": 1, "y": 2}})
        if model_type == MarksData
        else SoundMappingsData(mappings={"Lip Pop": "copy"})
    )

    protected = await protected_terms_validator.get_all_protected_terms()

    assert "copy" in protected
    assert app_config.grid.show_grid_phrase.lower() in protected
    assert app_config.mark.triggers.create_mark.lower() in protected
    assert app_config.dictation.start_trigger.lower() in protected
    assert {str(i) for i in range(1, 11)} <= protected
    assert "my mark" in protected
    assert "lip pop" in protected
    assert all(term == term.lower().strip() for term in protected)


@pytest.mark.asyncio
async def test_terms_cached_until_invalidated(protected_terms_validator, protected_terms_storage):
    first = await protected_terms_validator.get_all_protected_terms()
    reads = protected_terms_storage.read.call_count

    cached = await protected_terms_validator.get_all_protected_terms()
    assert cached == first
    assert protected_terms_storage.read.call_count == reads

    protected_terms_validator.invalidate_cache()
    await protected_terms_validator.get_all_protected_terms()
    assert protected_terms_storage.read.call_count > reads


@pytest.mark.asyncio
async def test_get_all_protected_terms_propagates_storage_error(protected_terms_validator, protected_terms_storage):
    protected_terms_storage.read.side_effect = Exception("Storage error")

    with pytest.raises(Exception, match="Storage error"):
        await protected_terms_validator.get_all_protected_terms()


@pytest.mark.parametrize("term,expected", [("copy", True), ("COPY", True), ("Copy", True), ("my custom command", False)])
@pytest.mark.asyncio
async def test_is_term_protected_is_case_insensitive(protected_terms_validator, term, expected):
    assert await protected_terms_validator.is_term_protected(term) is expected


@pytest.mark.parametrize("term", ["", "   "])
@pytest.mark.asyncio
async def test_validate_term_rejects_blank(protected_terms_validator, term):
    assert await protected_terms_validator.validate_term(term) == (False, "Term cannot be empty")


@pytest.mark.asyncio
async def test_validate_term_rejects_protected_term(protected_terms_validator):
    is_valid, error = await protected_terms_validator.validate_term("copy")
    assert is_valid is False
    assert "protected term" in error.lower()


@pytest.mark.asyncio
async def test_validate_term_accepts_unprotected_term(protected_terms_validator):
    assert await protected_terms_validator.validate_term("totally novel phrase") == (True, None)


@pytest.mark.asyncio
async def test_validate_term_allows_excluded_self(protected_terms_validator, protected_terms_storage):
    protected_terms_storage.read.side_effect = lambda model_type: (
        MarksData(marks={"my mark": {"x": 1, "y": 2}}) if model_type == MarksData else SoundMappingsData(mappings={})
    )

    assert (await protected_terms_validator.validate_term("my mark"))[0] is False
    assert await protected_terms_validator.validate_term("my mark", exclude_term="my mark") == (True, None)


@pytest.mark.asyncio
async def test_successful_mapping_update_invalidates_cache(event_bus, app_config, protected_terms_storage):
    validator = ProtectedTermsValidator(config=app_config, storage=protected_terms_storage)
    validator.setup_invalidation_subscriptions(event_bus)

    await validator.get_all_protected_terms()
    reads = protected_terms_storage.read.call_count

    await event_bus.publish(CommandMappingsUpdatedEvent(success=False, message="noop"))
    await asyncio.sleep(0.05)
    await validator.get_all_protected_terms()
    assert protected_terms_storage.read.call_count == reads

    await event_bus.publish(CommandMappingsUpdatedEvent(success=True, message="ok"))
    await asyncio.sleep(0.05)
    await validator.get_all_protected_terms()
    assert protected_terms_storage.read.call_count > reads
