from vocalance.app.ui.qt_theme import theme
from vocalance.app.ui.style.builder import build_app_stylesheet


def test_build_app_stylesheet_non_empty():
    qss = build_app_stylesheet(theme)
    assert len(qss) > 100
    assert "QComboBox" in qss
    assert "VocalanceScrollArea" in qss
    assert "{{" not in qss
