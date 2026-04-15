from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vocalance.app.ui.qt_theme import ThemeManager

_STYLE_DIR = Path(__file__).resolve().parent
_QSS_ORDER = ("base.qss", "native_controls.qss", "scrollable.qss")


def _token_map(theme: "ThemeManager") -> dict[str, str]:
    c = theme.config
    return {
        "FONT_PRIMARY": c.font_family_primary,
        "FONT_DISPLAY": c.font_family_display,
        "SHAPES_DARKEST": c.shapes.darkest,
        "SHAPES_DARK": c.shapes.dark,
        "SHAPES_MEDIUM": c.shapes.medium,
        "SHAPES_LIGHT": c.shapes.light,
        "SHAPES_LIGHTEST": c.shapes.lightest,
        "TEXT_LIGHTEST": c.text.lightest,
        "TEXT_LIGHT": c.text.light,
        "TEXT_MEDIUM": c.text.medium,
        "BLUE_2": c.blue.blue_2,
        "RADIUS_SMALL": str(c.radius.small),
        "SPACING_SMALL": str(c.spacing.small),
    }


def _apply_tokens(qss: str, tokens: dict[str, str]) -> str:
    out = qss
    for key, value in tokens.items():
        out = out.replace("{{" + key + "}}", value)
    return out


def build_app_stylesheet(theme: "ThemeManager") -> str:
    """Concatenate packaged QSS partials and inject theme tokens."""
    tokens = _token_map(theme)
    chunks: list[str] = []
    qss_dir = _STYLE_DIR / "qss"
    for name in _QSS_ORDER:
        path = qss_dir / name
        if path.is_file():
            raw = path.read_text(encoding="utf-8")
            chunks.append(_apply_tokens(raw, tokens))
    legacy = Path(__file__).resolve().parent.parent / "styles.qss"
    if legacy.is_file():
        chunks.append(_apply_tokens(legacy.read_text(encoding="utf-8"), tokens))
    return "\n\n".join(part.strip() for part in chunks if part.strip())
