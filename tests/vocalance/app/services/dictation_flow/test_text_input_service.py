import pytest
from conftest import skip_if_headless

skip_if_headless()


@pytest.mark.asyncio
async def test_input_text_stores_content_without_trailing_space(dictation_text_input, patched_keyboard):
    result = await dictation_text_input.input_text(text="hello world", add_trailing_space=True)

    assert result is True
    assert dictation_text_input.last_text == "hello world"


@pytest.mark.asyncio
async def test_input_text_presses_space_after_paste(dictation_text_input, patched_keyboard):
    _, press = patched_keyboard

    await dictation_text_input.input_text(text="hello", add_trailing_space=True)

    spaces = [c for c in press.call_args_list if c.args and c.args[0] == "space"]
    assert len(spaces) == 1


@pytest.mark.asyncio
async def test_input_text_no_space_when_add_trailing_space_false(dictation_text_input, patched_keyboard):
    _, press = patched_keyboard

    await dictation_text_input.input_text(text="hello", add_trailing_space=False)

    spaces = [c for c in press.call_args_list if c.args and c.args[0] == "space"]
    assert len(spaces) == 0


@pytest.mark.asyncio
async def test_input_text_empty_is_rejected(dictation_text_input, patched_keyboard):
    write, press = patched_keyboard

    result = await dictation_text_input.input_text(text="", add_trailing_space=True)

    assert result is False
    assert write.call_count == 0
    assert press.call_count == 0


@pytest.mark.asyncio
async def test_input_text_adds_period_when_new_segment_starts_with_capital(dictation_text_input, patched_keyboard):
    _, press = patched_keyboard
    dictation_text_input.last_text = "hello world"

    result = await dictation_text_input.input_text(text="Today is nice", add_trailing_space=True)

    assert result is True
    backspaces = [c for c in press.call_args_list if c.args and c.args[0] == "backspace"]
    assert len(backspaces) == 1


@pytest.mark.parametrize("terminal_char", [".", "?", "!"])
@pytest.mark.asyncio
async def test_input_text_no_period_when_previous_ended_with_terminal_punctuation(
    terminal_char, dictation_text_input, patched_keyboard
):
    _, press = patched_keyboard
    dictation_text_input.last_text = f"hello world{terminal_char}"

    await dictation_text_input.input_text(text="Today is nice", add_trailing_space=True)

    backspaces = [c for c in press.call_args_list if c.args and c.args[0] == "backspace"]
    assert len(backspaces) == 0


@pytest.mark.asyncio
async def test_input_text_no_period_when_new_segment_starts_with_lowercase(dictation_text_input, patched_keyboard):
    _, press = patched_keyboard
    dictation_text_input.last_text = "hello world"

    await dictation_text_input.input_text(text="and more text", add_trailing_space=True)

    backspaces = [c for c in press.call_args_list if c.args and c.args[0] == "backspace"]
    assert len(backspaces) == 0


@pytest.mark.parametrize("first_word", ["I", "NASA", "CEO", "A", "OK"])
@pytest.mark.asyncio
async def test_input_text_no_period_when_first_word_is_all_caps(first_word, dictation_text_input, patched_keyboard):
    _, press = patched_keyboard
    dictation_text_input.last_text = "hello world"

    await dictation_text_input.input_text(text=f"{first_word} am going now", add_trailing_space=True)

    backspaces = [c for c in press.call_args_list if c.args and c.args[0] == "backspace"]
    assert len(backspaces) == 0
