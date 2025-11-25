"""UI Components module.

Provides reusable Qt component subclasses with consistent styling.
"""

# Buttons
from vocalance.app.ui.components.buttons import DangerButton, GhostButton, PrimaryButton

# Checkboxes
from vocalance.app.ui.components.checkboxes import Checkbox

# Complex components
from vocalance.app.ui.components.complex_components import FormGroup, SidebarButton, Tile

# Dialogs
from vocalance.app.ui.components.dialogs import (
    BaseDialog,
    CommandEditDialog,
    PromptEditDialog,
    askokcancel,
    askyesno,
    showerror,
    showinfo,
    showwarning,
)

# Inputs
from vocalance.app.ui.components.inputs import PasswordInput, TextInput

# Labels
from vocalance.app.ui.components.labels import (
    BodyLabel,
    BoxTitleLabel,
    GroupHeaderLabel,
    LargeLabel,
    SmallLabel,
    SubtitleLabel,
    TitleLabel,
)

# Layouts
from vocalance.app.ui.components.layouts import (
    BaseContainer,
    Box,
    Card,
    ContentArea,
    FormField,
    GroupHeader,
    ListForm,
    ListItem,
    Panel,
    ScrollableContainer,
    TransparentBox,
    TransparentViewport,
    TransparentWidget,
    TwoColumnLayout,
)

__all__ = [
    # Labels
    "TitleLabel",
    "SubtitleLabel",
    "BodyLabel",
    "SmallLabel",
    "GroupHeaderLabel",
    "BoxTitleLabel",
    "LargeLabel",
    # Buttons
    "PrimaryButton",
    "DangerButton",
    "GhostButton",
    # Inputs
    "TextInput",
    "PasswordInput",
    # Checkboxes
    "Checkbox",
    # Dialogs
    "BaseDialog",
    "CommandEditDialog",
    "PromptEditDialog",
    "askokcancel",
    "askyesno",
    "showinfo",
    "showerror",
    "showwarning",
    # Layouts
    "BaseContainer",
    "Box",
    "Panel",
    "Card",
    "TransparentBox",
    "TransparentWidget",
    "TransparentViewport",
    "ContentArea",
    "ScrollableContainer",
    "TwoColumnLayout",
    "FormField",
    "ListItem",
    "GroupHeader",
    "ListForm",
    # Complex components
    "FormGroup",
    "Tile",
    "SidebarButton",
]
