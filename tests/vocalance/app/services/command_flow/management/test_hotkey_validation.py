import pytest

from vocalance.app.config.hotkey_validation import is_allowed_key, validate_custom_hotkey


class TestIsAllowedKey:
    def test_single_letter(self):
        assert is_allowed_key("a") is True

    def test_single_digit(self):
        assert is_allowed_key("5") is True

    def test_modifier_keys(self):
        for key in ("ctrl", "alt", "shift", "win", "cmd", "fn"):
            assert is_allowed_key(key) is True

    def test_named_keys(self):
        for key in ("enter", "tab", "space", "backspace", "escape", "up", "down", "left", "right", "f1", "f12", "f24"):
            assert is_allowed_key(key) is True

    def test_symbol_keys(self):
        for key in ("-", "=", "/", ".", "'", "[", "]"):
            assert is_allowed_key(key) is True

    def test_empty_string_rejected(self):
        assert is_allowed_key("") is False

    def test_whitespace_only_rejected(self):
        assert is_allowed_key("   ") is False

    def test_invalid_multi_char_rejected(self):
        assert is_allowed_key("custom") is False
        assert is_allowed_key("macro") is False

    def test_f_key_out_of_range_rejected(self):
        assert is_allowed_key("f0") is False
        assert is_allowed_key("f25") is False

    def test_case_insensitive(self):
        assert is_allowed_key("CTRL") is True
        assert is_allowed_key("Enter") is True
        assert is_allowed_key("F5") is True


class TestValidateCustomHotkey:
    def test_single_key(self):
        assert validate_custom_hotkey("a") is None

    def test_simple_chord(self):
        assert validate_custom_hotkey("ctrl+s") is None

    def test_multi_modifier_chord(self):
        assert validate_custom_hotkey("ctrl+shift+k") is None

    def test_modifier_with_function_key(self):
        assert validate_custom_hotkey("alt+f4") is None
        assert validate_custom_hotkey("shift+alt+f5") is None

    def test_modifier_with_named_key(self):
        assert validate_custom_hotkey("ctrl+enter") is None
        assert validate_custom_hotkey("ctrl+backspace") is None

    def test_spaces_around_plus_accepted(self):
        assert validate_custom_hotkey("ctrl + s") is None

    def test_empty_string_rejected(self):
        assert validate_custom_hotkey("") is not None

    def test_whitespace_only_rejected(self):
        assert validate_custom_hotkey("   ") is not None

    def test_comma_separated_sequences_rejected(self):
        assert validate_custom_hotkey("ctrl+c, ctrl+v") is not None
        assert validate_custom_hotkey("a,b") is not None

    def test_semicolon_separated_rejected(self):
        assert validate_custom_hotkey("ctrl+c;ctrl+v") is not None

    def test_unknown_key_token_rejected(self):
        assert validate_custom_hotkey("ctrl+custom") is not None
        assert validate_custom_hotkey("macro") is not None

    def test_error_message_is_string(self):
        error = validate_custom_hotkey("ctrl+custom")
        assert isinstance(error, str)
        assert len(error) > 0

    def test_valid_returns_none(self):
        assert validate_custom_hotkey("ctrl+alt+7") is None


class TestAutomationCommandModelValidation:
    def test_valid_custom_hotkey_accepted(self):
        from vocalance.app.config.command_types import AutomationCommand

        cmd = AutomationCommand(command_key="test", action_type="hotkey", action_value="ctrl+k", is_custom=True)
        assert cmd.action_value == "ctrl+k"

    def test_invalid_custom_hotkey_rejected(self):
        from pydantic import ValidationError

        from vocalance.app.config.command_types import AutomationCommand

        with pytest.raises(ValidationError):
            AutomationCommand(command_key="test", action_type="hotkey", action_value="ctrl+custom", is_custom=True)

    def test_comma_sequence_custom_rejected(self):
        from pydantic import ValidationError

        from vocalance.app.config.command_types import AutomationCommand

        with pytest.raises(ValidationError):
            AutomationCommand(command_key="test", action_type="hotkey", action_value="ctrl+c,ctrl+v", is_custom=True)

    def test_builtin_bypasses_validation(self):
        from vocalance.app.config.command_types import AutomationCommand

        cmd = AutomationCommand(command_key="min", action_type="key_sequence", action_value="alt+space, n", is_custom=False)
        assert cmd.action_value == "alt+space, n"

    def test_builtin_hotkey_with_unusual_value_accepted(self):
        from vocalance.app.config.command_types import AutomationCommand

        cmd = AutomationCommand(command_key="close all", action_type="hotkey", action_value="ctrl+m+w", is_custom=False)
        assert cmd.action_value == "ctrl+m+w"
