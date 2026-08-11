from conftest import skip_if_headless

skip_if_headless()

from vocalance.app.ui.style.builder import build_app_stylesheet, collect_theme_tokens, inject_tokens_into_qss


def test_inject_tokens_substitutes_all_occurrences():
    qss = "QWidget { color: {{C}}; border-color: {{C}}; font: {{F}}; }"
    result = inject_tokens_into_qss(qss, {"C": "#fff", "F": "DM Sans"})
    assert result == "QWidget { color: #fff; border-color: #fff; font: DM Sans; }"
    assert "{{" not in result


def test_inject_tokens_leaves_unknown_placeholders_untouched():
    result = inject_tokens_into_qss("a {{KNOWN}} b {{MISSING}}", {"KNOWN": "x"})
    assert result == "a x b {{MISSING}}"


def test_collect_theme_tokens_values_are_strings(theme_manager):
    tokens = collect_theme_tokens(theme_manager)
    assert all(isinstance(value, str) for value in tokens.values())


def test_collect_theme_tokens_maps_config_values(theme_manager):
    tokens = collect_theme_tokens(theme_manager)
    config = theme_manager.config
    assert tokens["FONT_PRIMARY"] == config.font_family_primary
    assert tokens["SHAPES_DARKEST"] == config.shapes.darkest
    assert tokens["RADIUS_SMALL"] == str(config.radius.small)


def test_build_app_stylesheet_composes_and_substitutes(theme_manager):
    qss = build_app_stylesheet(theme_manager)
    assert "QComboBox" in qss
    assert "VocalanceScrollArea" in qss
    assert "{{" not in qss
    assert theme_manager.config.shapes.darkest in qss
