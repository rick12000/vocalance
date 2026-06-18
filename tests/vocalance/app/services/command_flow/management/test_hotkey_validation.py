import pytest

from vocalance.app.config.hotkey_validation import is_allowed_key, validate_custom_hotkey


@pytest.mark.parametrize(
    "token",
    [
        "a",
        "5",
        "ctrl",
        "alt",
        "shift",
        "win",
        "cmd",
        "fn",
        "enter",
        "tab",
        "space",
        "f1",
        "f12",
        "f24",
        "-",
        "/",
        "[",
        "CTRL",
        "Enter",
        "F5",
    ],
)
def test_is_allowed_key_accepts_recognised_tokens(token):
    assert is_allowed_key(token) is True


@pytest.mark.parametrize("token", ["", "   ", "custom", "macro", "f0", "f25", "ab"])
def test_is_allowed_key_rejects_unrecognised_tokens(token):
    assert is_allowed_key(token) is False


@pytest.mark.parametrize(
    "value",
    ["a", "ctrl+s", "ctrl+shift+k", "alt+f4", "ctrl+enter", "ctrl + s", "ctrl+alt+7"],
)
def test_validate_custom_hotkey_accepts_single_chord(value):
    assert validate_custom_hotkey(value) is None


@pytest.mark.parametrize(
    "value",
    ["", "   ", "ctrl+c, ctrl+v", "a,b", "ctrl+c;ctrl+v", "ctrl+custom", "macro"],
)
def test_validate_custom_hotkey_rejects_invalid(value):
    error = validate_custom_hotkey(value)
    assert isinstance(error, str)
    assert error
