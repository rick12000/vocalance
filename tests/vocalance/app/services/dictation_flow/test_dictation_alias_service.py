import pytest


@pytest.mark.asyncio
async def test_add_alias_persists_new_entry(dictation_alias_service):
    result = await dictation_alias_service.add_alias("greeting", "hello there")

    assert result is True
    assert dictation_alias_service.get_aliases()["greeting"] == "hello there"
    dictation_alias_service.storage.write.assert_called_once()


@pytest.mark.asyncio
async def test_add_alias_normalizes_key_and_value(dictation_alias_service):
    await dictation_alias_service.add_alias("  GREETING  ", "  hello world  ")

    assert dictation_alias_service.get_aliases()["greeting"] == "hello world"


@pytest.mark.asyncio
async def test_add_alias_duplicate_fails(dictation_alias_service):
    await dictation_alias_service.add_alias("greeting", "hello")

    result = await dictation_alias_service.add_alias("greeting", "hi")

    assert result is False
    assert dictation_alias_service.get_aliases()["greeting"] == "hello"


@pytest.mark.parametrize("key,value", [("", "value"), ("key", "")])
@pytest.mark.asyncio
async def test_add_alias_rejects_empty(dictation_alias_service, key, value):
    assert await dictation_alias_service.add_alias(key, value) is False


@pytest.mark.asyncio
async def test_add_alias_rolls_back_on_save_failure(dictation_alias_service):
    dictation_alias_service.storage.write.return_value = False

    result = await dictation_alias_service.add_alias("greeting", "hello")

    assert result is False
    assert "greeting" not in dictation_alias_service.get_aliases()


@pytest.mark.asyncio
async def test_update_alias_replaces_value(dictation_alias_service):
    await dictation_alias_service.add_alias("greeting", "hello")

    result = await dictation_alias_service.update_alias("greeting", "hi there")

    assert result is True
    assert dictation_alias_service.get_aliases()["greeting"] == "hi there"


@pytest.mark.asyncio
async def test_update_alias_nonexistent_fails(dictation_alias_service):
    assert await dictation_alias_service.update_alias("nonexistent", "value") is False


@pytest.mark.asyncio
async def test_update_alias_rolls_back_on_save_failure(dictation_alias_service):
    await dictation_alias_service.add_alias("greeting", "hello")
    dictation_alias_service.storage.write.return_value = False

    result = await dictation_alias_service.update_alias("greeting", "hi")

    assert result is False
    assert dictation_alias_service.get_aliases()["greeting"] == "hello"


@pytest.mark.asyncio
async def test_delete_alias_removes_entry(dictation_alias_service):
    await dictation_alias_service.add_alias("greeting", "hello")

    result = await dictation_alias_service.delete_alias("greeting")

    assert result is True
    assert "greeting" not in dictation_alias_service.get_aliases()


@pytest.mark.asyncio
async def test_delete_alias_nonexistent_fails(dictation_alias_service):
    assert await dictation_alias_service.delete_alias("nonexistent") is False


@pytest.mark.asyncio
async def test_delete_alias_rolls_back_on_save_failure(dictation_alias_service):
    await dictation_alias_service.add_alias("greeting", "hello")
    dictation_alias_service.storage.write.return_value = False

    result = await dictation_alias_service.delete_alias("greeting")

    assert result is False
    assert "greeting" in dictation_alias_service.get_aliases()


@pytest.mark.parametrize(
    "aliases,text,expected",
    [
        ({"greeting": "hello world"}, "insert greeting to everyone", "hello world to everyone"),
        ({"greeting": "hello"}, "Insert Greeting to all", "hello to all"),
        ({"greeting": "hello", "farewell": "goodbye"}, "insert greeting and insert farewell", "hello and goodbye"),
        ({"test": "value", "test phrase": "phrase_value"}, "insert test phrase", "phrase_value"),
        ({"test": "value"}, "insert test and testing", "value and testing"),
        ({"test": "value"}, "insert something", "insert something"),
        ({"greeting": "hi"}, "", ""),
    ],
)
@pytest.mark.asyncio
async def test_apply_substitutions(dictation_alias_service, aliases, text, expected):
    dictation_alias_service.aliases = aliases
    assert dictation_alias_service.apply_substitutions(text) == expected


@pytest.mark.asyncio
async def test_extract_aliases_replaces_match_with_placeholder(dictation_alias_service):
    dictation_alias_service.aliases = {"greeting": "hello world"}

    text_with_placeholders, alias_map = dictation_alias_service.extract_aliases("please insert greeting now")

    assert "insert greeting" not in text_with_placeholders
    assert len(alias_map) == 1
    placeholder, substitution = next(iter(alias_map.items()))
    assert placeholder in text_with_placeholders
    assert substitution == "hello world"


@pytest.mark.asyncio
async def test_extract_aliases_without_aliases_is_identity(dictation_alias_service):
    text, alias_map = dictation_alias_service.extract_aliases("nothing to extract here")

    assert text == "nothing to extract here"
    assert alias_map == {}
