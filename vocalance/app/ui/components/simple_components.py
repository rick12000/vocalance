"""Simple UI components built from primitives with variant systems.

Components wrap primitives and add variant logic programmatically.
NO STYLESHEETS - only programmatic configuration via primitives.
"""

from typing import Literal, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from vocalance.app.ui.components.primitives import PrimitiveButton, PrimitiveCheckbox, PrimitiveInput, PrimitiveLabel
from vocalance.app.ui.qt_theme import theme


class Label(PrimitiveLabel):
    """Label component with variant-based styling.

    Variants:
    - title: Large, bold, lightest color
    - subtitle: Large, semibold, light color
    - body: Medium, regular, lightest color
    - small: Small, regular, light color
    - group_header: Medium, semibold, medium color
    - box_title: Large, semibold, lightest color
    - large: Large size (for special cases)
    """

    def __init__(
        self,
        text: str = "",
        parent: Optional[QWidget] = None,
        variant: Literal["title", "subtitle", "body", "small", "group_header", "box_title", "large"] = "body",
        color: Optional[str] = None,
        align: Literal["left", "center", "right"] = "left",
    ):
        # Determine font and color based on variant
        if variant == "title":
            font_size = theme.config.fonts.xxlarge
            font_weight = "bold"
            default_color = theme.config.text.lightest
        elif variant == "subtitle":
            font_size = theme.config.fonts.large
            font_weight = "semibold"
            default_color = theme.config.text.light
        elif variant == "body":
            font_size = theme.config.fonts.medium
            font_weight = "regular"
            default_color = theme.config.text.lightest
        elif variant == "small":
            font_size = theme.config.fonts.small
            font_weight = "regular"
            default_color = theme.config.text.light
        elif variant == "group_header":
            font_size = theme.config.fonts.medium
            font_weight = "semibold"
            default_color = theme.config.text.medium
        elif variant == "box_title":
            font_size = theme.config.fonts.large
            font_weight = "semibold"
            default_color = theme.config.text.lightest
        elif variant == "large":
            font_size = theme.config.fonts.large
            font_weight = "regular"
            default_color = theme.config.text.lightest
        else:
            font_size = theme.config.fonts.medium
            font_weight = "regular"
            default_color = theme.config.text.lightest

        # Use provided color or default
        final_color = color if color else default_color

        # Initialize with configuration
        super().__init__(
            text=text,
            parent=parent,
            font_size=font_size,
            font_weight=font_weight,
            color=final_color,
        )

        # Set alignment
        align_map = {
            "left": Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            "center": Qt.AlignmentFlag.AlignCenter,
            "right": Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        }
        self.setAlignment(align_map.get(align, Qt.AlignmentFlag.AlignLeft))


class Button(PrimitiveButton):
    """Button component with variant-based styling.

    Variants:
    - primary: Accent background, dark text
    - danger: Medium background with light border, lightest text
    - ghost: Transparent background, light text
    """

    def __init__(
        self,
        text: str = "",
        parent: Optional[QWidget] = None,
        variant: Literal["primary", "danger", "ghost"] = "primary",
        icon=None,
        command=None,
    ):
        # Determine colors based on variant
        if variant == "primary":
            bg_color = theme.config.shapes.accent
            text_color = theme.config.text.light_blue_accent
        elif variant == "danger":
            bg_color = theme.config.shapes.medium
            text_color = theme.config.text.lightest
        elif variant == "ghost":
            bg_color = "transparent"
            text_color = theme.config.text.light
        else:
            bg_color = theme.config.shapes.accent
            text_color = theme.config.text.light_blue_accent

        # Store variant for later use
        self._variant = variant

        # Initialize with configuration
        super().__init__(
            text=text,
            parent=parent,
            bg_color=bg_color,
            text_color=text_color,
            height=theme.config.components.button_height,
        )

        # Override variant-specific hover colors after initialization
        if variant == "danger":
            self._hover_bg_color = theme.config.shapes.light
            self._pressed_bg_color = theme.config.shapes.medium
        elif variant == "ghost":
            self._hover_bg_color = theme.config.shapes.medium
            self._pressed_bg_color = theme.config.shapes.dark

        # Set icon if provided
        if icon:
            self.setIcon(icon)

        # Connect command if provided
        if command:
            self.clicked.connect(command)


class Input(PrimitiveInput):
    """Text input component.

    Wraps PrimitiveInput with no additional configuration.
    """

    def __init__(
        self,
        placeholder: str = "",
        parent: Optional[QWidget] = None,
        password: bool = False,
    ):
        super().__init__(placeholder=placeholder, parent=parent)

        if password:
            self.setEchoMode(self.EchoMode.Password)


class Checkbox(PrimitiveCheckbox):
    """Checkbox component.

    Wraps PrimitiveCheckbox with optional command callback.
    """

    def __init__(
        self,
        text: str = "",
        parent: Optional[QWidget] = None,
        checked: bool = False,
        command=None,
    ):
        super().__init__(text=text, parent=parent)

        self.setChecked(checked)

        if command:
            self.stateChanged.connect(command)
