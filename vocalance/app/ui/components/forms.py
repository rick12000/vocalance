from typing import Optional, Tuple

from PySide6.QtWidgets import QVBoxLayout, QWidget

from vocalance.app.ui.qt_theme import theme

from .atoms import Input, Label


class FormGroup(QWidget):
    """Container for a label and an input field."""

    def __init__(self, label: str, input_widget: QWidget, parent: Optional[QWidget] = None, description: Optional[str] = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.config.spacing.tiny)

        # Label
        self.label = Label(label, variant="small", color="text.light")
        layout.addWidget(self.label)

        # Input
        layout.addWidget(input_widget)

        # Optional description
        if description:
            desc = Label(description, variant="small", color="text.medium")
            layout.addWidget(desc)

    @staticmethod
    def create_text(
        label: str, placeholder: str = "", default: str = "", parent: Optional[QWidget] = None
    ) -> Tuple["FormGroup", Input]:
        """Factory to create a text input form group."""
        inp = Input(placeholder)
        if default:
            inp.setText(str(default))
        group = FormGroup(label, inp, parent)
        return group, inp
