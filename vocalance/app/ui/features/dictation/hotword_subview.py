import logging
from typing import Optional

from PySide6.QtGui import QTextOption
from PySide6.QtWidgets import QMessageBox, QVBoxLayout, QWidget

from vocalance.app.ui.components.buttons import PrimaryButton
from vocalance.app.ui.components.inputs import ExpandableTextArea
from vocalance.app.ui.components.layouts import TransparentBox, TwoColumnLayout
from vocalance.app.ui.components.tile import Tile
from vocalance.app.ui.qt_theme import theme


class QtHotwordSubView(QWidget):
    """Hotword editor: instructions on the left, one word per line on the right."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.controller = None
        self.setup_ui()

    def set_controller(self, controller) -> None:
        """Wire signals and request the saved hotword list."""
        self.controller = controller
        self.controller.hotwords_loaded.connect(self.on_hotwords_loaded)
        self.controller.operation_error.connect(self.show_error)
        self.controller.refresh_hotwords()

    def setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.layout_widget = TwoColumnLayout("Instructions", "Hot Words", self)
        main_layout.addWidget(self.layout_widget, stretch=1)
        self.setup_instructions_panel()
        self.setup_editor_panel()

    def setup_instructions_panel(self) -> None:
        content = self.layout_widget.left_content
        content.add(
            Tile(
                "Bias a spelling",
                "A hotword makes that spelling more likely,\nincluding a name the model has never seen,\nwhen the pronunciation fits it.",
            ),
            stretch=1,
        )
        content.add(
            Tile(
                "One per line",
                "No spaces. Case is kept, so API and Api\nare different entries.",
            ),
            stretch=1,
        )
        content.add(
            Tile(
                "When it applies",
                "Save rejects a bad list and keeps the old one.\nA saved list is used the next time dictation starts.\nUntil the spelling file has downloaded, the list is kept but not applied.",
            ),
            stretch=1,
        )

    def setup_editor_panel(self) -> None:
        content = self.layout_widget.right_content
        self.editor = ExpandableTextArea(placeholder="One hotword per line")
        self.editor.setWordWrapMode(QTextOption.WrapMode.NoWrap)
        self.editor._min_height_lines = 12
        self.editor._calculate_height_from_placeholder()
        content.add(self.editor, stretch=1)

        button_box = TransparentBox(layout="horizontal", spacing=theme.config.spacing.small)
        button_box.add(PrimaryButton(text="Save", command=self.on_save_clicked))
        content.add(button_box)

    def on_hotwords_loaded(self, hotwords: list) -> None:
        self.editor.setText("\n".join(hotwords))
        self.logger.info("Hotwords loaded: %s total", len(hotwords))

    def on_save_clicked(self) -> None:
        if self.controller:
            self.controller.save_hotwords(self.editor.text())

    def show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Hot Words", message)
