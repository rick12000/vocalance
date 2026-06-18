import pytest


@pytest.mark.asyncio
async def test_input_text_types_with_trailing_space(dictation_text_input, patched_keyboard):
    write, _ = patched_keyboard

    result = await dictation_text_input.input_text("hello world")

    assert result is True
    assert dictation_text_input.last_text == "hello world "
    assert write.called


@pytest.mark.asyncio
async def test_input_text_empty_is_rejected(dictation_text_input, patched_keyboard):
    write, press = patched_keyboard

    result = await dictation_text_input.input_text("")

    assert result is False
    assert write.call_count == 0
    assert press.call_count == 0


@pytest.mark.asyncio
async def test_input_text_removes_previous_period_and_prefixes_space(dictation_text_input, patched_keyboard):
    _, press = patched_keyboard
    dictation_text_input.last_text = "Previous sentence. "

    result = await dictation_text_input.input_text("lowercase continuation")

    assert result is True
    assert dictation_text_input.last_text == " lowercase continuation "
    backspaces = sum(1 for call in press.call_args_list if call.args and call.args[0] == "backspace")
    assert backspaces == 2


@pytest.mark.asyncio
async def test_input_text_lowercases_mid_sentence_continuation(dictation_text_input, patched_keyboard):
    dictation_text_input.last_text = "No sentence boundary "

    result = await dictation_text_input.input_text("Uppercase start")

    assert result is True
    assert dictation_text_input.last_text == "uppercase start "


@pytest.mark.asyncio
async def test_input_text_skip_join_preserves_identifier_casing(dictation_text_input, patched_keyboard):
    dictation_text_input.last_text = "prior chunk "

    result = await dictation_text_input.input_text("HelloWorld", skip_prose_segment_join_rules=True, add_trailing_space=False)

    assert result is True
    assert dictation_text_input.last_text == "HelloWorld"
