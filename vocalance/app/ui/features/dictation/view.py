import logging
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QHBoxLayout, QMessageBox, QPushButton, QRadioButton, QStackedWidget, QVBoxLayout, QWidget

from vocalance.app.ui.components.buttons import DeleteButton, PrimaryButton
from vocalance.app.ui.components.dialogs import PromptEditDialog
from vocalance.app.ui.components.inputs import ExpandableTextArea, TextInput
from vocalance.app.ui.components.labels import BodyLabel, SmallLabel
from vocalance.app.ui.components.layouts import (
    BaseContainer,
    ScrollableContainer,
    TransparentBox,
    TransparentWidget,
    TwoColumnLayout,
)
from vocalance.app.ui.features.dictation.alias_subview import QtDictationAliasSubView
from vocalance.app.ui.qt_theme import theme
from vocalance.app.utils.llm_dep_check import llm_deps_available


class TabButton(QPushButton):
    """Styled pill-shaped tab button for horizontal menu navigation."""

    def __init__(self, text: str, parent: Optional[QWidget] = None):
        super().__init__(text, parent)
        self.chosen = False
        self.setMinimumHeight(32)
        self.setMinimumWidth(90)

        # Set font directly - use display font with medium size
        self.setFont(theme.get_font(size="medium", weight="semibold", display=True))

        self.apply_style()

    def apply_style(self) -> None:
        """Apply styling based on selection state - pill-shaped buttons."""
        if self.chosen:
            # Selected state: dark background with full styling
            self.setStyleSheet(
                f"""
                TabButton {{
                    background-color: {theme.config.shapes.dark};
                    color: {theme.config.text.light};
                    border: none;
                    border-radius: 16px;
                    padding: 4px 16px;
                }}
                TabButton:hover {{
                    background-color: {theme.config.shapes.dark};
                }}
                """
            )
        else:
            # Unselected state: transparent background
            self.setStyleSheet(
                f"""
                TabButton {{
                    background-color: transparent;
                    color: {theme.config.text.medium};
                    border: none;
                    border-radius: 16px;
                    padding: 4px 16px;
                }}
                TabButton:hover {{
                    background-color: {theme.config.shapes.dark};
                    color: {theme.config.text.light};
                }}
                """
            )

    def set_selected(self, selected: bool) -> None:
        """Set the selection state of the button."""
        self.chosen = selected
        self.apply_style()

    def is_selected(self) -> bool:
        """Check if the button is selected."""
        return self.chosen


class QtPromptsSubView(QWidget):
    """Sub-view for managing dictation prompts.

    This is the original prompts management functionality extracted as a sub-view.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize prompts sub-view."""
        super().__init__(parent)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.controller = None
        self.prompts_list: List[Dict[str, Any]] = []
        self.current_prompt_id = None
        self.prompt_radio_buttons = {}

        self.setup_ui()

    def set_controller(self, controller) -> None:
        self.controller = controller

        self.controller.prompts_loaded.connect(self.on_prompts_loaded)
        self.controller.current_prompt_updated.connect(self.on_current_prompt_updated)
        self.controller.operation_error.connect(self.on_error)
        self.controller.status_updated.connect(self.on_status_updated)

        self.logger.info("Loading prompts from controller")
        self.controller.refresh_prompts()

    def setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.layout_widget = TwoColumnLayout("Add Custom Prompt", "Manage Prompts", self)
        main_layout.addWidget(self.layout_widget, stretch=1)

        # Setup panels
        self.setup_add_prompt_form()
        self.setup_manage_prompts_panel()

    def setup_add_prompt_form(self) -> None:
        content = self.layout_widget.left_content

        # Prompt title input
        prompt_title_label = BodyLabel("Prompt Title:")
        content.add(prompt_title_label)
        self.title_entry = TextInput(placeholder="e.g. Email Formatting")
        content.add(self.title_entry)

        # Prompt instructions
        prompt_instructions_label = BodyLabel("Prompt Instructions:")
        content.add(prompt_instructions_label)
        self.prompt_textbox = ExpandableTextArea(
            placeholder="e.g. Format as an email. Start with 'Dear [Recipient Name],' and end with 'Best, Jim.' Adopt a professional tone and style."
        )
        content.add(self.prompt_textbox)

        # Add button
        self.add_prompt_btn = PrimaryButton(text="Add Prompt")
        self.add_prompt_btn.clicked.connect(self.on_add_prompt_clicked)
        content.add(self.add_prompt_btn)

        content.add_stretch()

    def setup_manage_prompts_panel(self) -> None:
        content = self.layout_widget.right_content

        # Prompts list widget
        self.prompts_list_widget = TransparentWidget()
        self.prompts_list_layout = QVBoxLayout(self.prompts_list_widget)
        self.prompts_list_layout.setSpacing(theme.config.container.list_item_spacing)
        self.prompts_list_layout.setContentsMargins(0, 0, theme.config.spacing.small, 0)

        # Scroll area for prompts
        scroll_area = ScrollableContainer()
        scroll_area.content_layout.addWidget(self.prompts_list_widget)
        content.add(scroll_area, stretch=1)

    def on_prompts_loaded(self, prompts: List[Dict[str, Any]]) -> None:
        self.prompts_list = prompts
        self.current_prompt_id = self.controller.get_current_prompt_id() if self.controller else None
        self.display_prompts(prompts)
        self.logger.info("Prompts loaded: %s total", len(prompts))

    def on_current_prompt_updated(self, prompt_id: str) -> None:
        self.current_prompt_id = prompt_id
        if prompt_id in self.prompt_radio_buttons:
            self.prompt_radio_buttons[prompt_id].setChecked(True)
        self.logger.info("Current prompt updated: %s", prompt_id)

    def on_status_updated(self, message: str, is_error: bool) -> None:
        if is_error:
            self.show_error(message)

    def on_error(self, error_msg: str) -> None:
        self.show_error(error_msg)

    def display_prompts(self, prompts: List[Dict[str, Any]]) -> None:
        # Clear existing items
        while self.prompts_list_layout.count():
            item = self.prompts_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.prompt_radio_buttons.clear()

        if not prompts:
            empty_label = BodyLabel("No prompts available.", align="center", color=theme.config.text.medium)
            self.prompts_list_layout.addWidget(empty_label)
        else:
            for prompt in prompts:
                self.create_prompt_item(prompt)

        self.prompts_list_layout.addStretch()

    def create_prompt_item(self, prompt_data: Dict[str, Any]) -> None:
        # Create item widget
        item_widget = TransparentWidget()
        item_widget.setProperty("itemType", "list_item")
        item_layout = QHBoxLayout(item_widget)
        item_layout.setContentsMargins(
            0,
            theme.config.container.list_item_padding_vertical,
            0,
            theme.config.container.list_item_padding_vertical,
        )
        item_layout.setSpacing(theme.config.spacing.small)

        # Radio button
        radio = QRadioButton()
        prompt_id = prompt_data.get("id", "")
        is_current = prompt_data.get("is_current", False) or (prompt_id == self.current_prompt_id)
        radio.setChecked(is_current)
        radio.toggled.connect(lambda checked, pid=prompt_id: self.on_radio_selected(pid, checked))
        self.prompt_radio_buttons[prompt_id] = radio
        item_layout.addWidget(radio)

        # Prompt name label
        name_label = SmallLabel(prompt_data.get("name", "Unnamed"), color=theme.config.text.medium)
        item_layout.addWidget(name_label, 1)

        # Edit button
        edit_btn = PrimaryButton(text="Edit")
        edit_btn.setFixedWidth(theme.config.components.button_action_width - 10)
        edit_btn.clicked.connect(lambda checked, p=prompt_data: self.on_edit_prompt(p))
        item_layout.addWidget(edit_btn)

        # Delete button
        is_protected = prompt_data.get("is_default", False) or bool(prompt_data.get("system_key"))
        delete_btn = DeleteButton(
            command=lambda checked, pid=prompt_data.get("id"): self.on_delete_prompt(pid) if not is_protected else None
        )
        delete_btn.setEnabled(not is_protected)
        item_layout.addWidget(delete_btn)

        self.prompts_list_layout.insertWidget(self.prompts_list_layout.count() - 1, item_widget)

    def on_radio_selected(self, prompt_id: str, checked: bool) -> None:
        if checked and self.controller:
            self.controller.select_prompt(prompt_id)

    def on_add_prompt_clicked(self) -> None:
        title = self.title_entry.text().strip()
        prompt_text = self.prompt_textbox.text().strip()

        if not title:
            QMessageBox.warning(self, "Validation Error", "Please enter a title for the prompt.")
            return

        if not prompt_text:
            QMessageBox.warning(self, "Validation Error", "Please enter prompt instructions.")
            return

        if not self.controller:
            QMessageBox.critical(self, "Error", "Controller not initialized.")
            return

        if self.controller.add_prompt(title, prompt_text):
            self.title_entry.clear()
            self.prompt_textbox.clear()

    def on_edit_prompt(self, prompt_data: Dict[str, Any]) -> None:
        if prompt_data.get("is_default", False):
            QMessageBox.information(self, "Cannot Edit Default Prompt", "The default prompt cannot be edited.")
            return

        if not self.controller:
            self.show_error("Controller not initialized.")
            return

        dialog = PromptEditDialog(prompt_data, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            if dialog.result_saved:
                self.controller.edit_prompt(prompt_data["id"], dialog.new_name, dialog.new_text)
                self.logger.info(f"Edited prompt: {prompt_data['id']}")

    def on_delete_prompt(self, prompt_id: str) -> None:
        if self.controller and self.controller.is_protected_prompt(prompt_id):
            QMessageBox.information(self, "Cannot Delete Prompt", "This prompt cannot be deleted.")
            return

        if self.controller:
            self.controller.delete_prompt(prompt_id)

    def show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Error", message)


class QtDictationView(QWidget):
    """Qt-based dictation view with tabbed sub-views.

    Features:
    - Horizontal menu with tab buttons (Prompts, Aliases)
    - Stacked widget to switch between sub-views
    - Prompts sub-view: Add/edit/delete/select prompts
    - Aliases sub-view: Add/edit/delete alias substitutions
    - Real-time updates from controllers
    """

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize dictation view."""
        super().__init__(parent)

        self.logger = logging.getLogger(self.__class__.__name__)
        self._llm_enabled = llm_deps_available()
        self.dictation_controller = None
        self.alias_controller = None
        self.tab_buttons: Dict[str, TabButton] = {}

        self.setup_main_ui()
        self.logger.debug("QtDictationView initialized")

    def set_controller(self, controller) -> None:
        """Set the dictation controller and connect to prompts sub-view."""
        self.dictation_controller = controller
        if self.prompts_sub_view is not None:
            self.prompts_sub_view.set_controller(controller)

    def set_alias_controller(self, controller) -> None:
        """Set the alias controller and connect to aliases sub-view."""
        self.alias_controller = controller
        self.aliases_sub_view.set_controller(controller)

    def setup_main_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.prompts_sub_view: Optional[QtPromptsSubView] = QtPromptsSubView() if self._llm_enabled else None
        self.aliases_sub_view = QtDictationAliasSubView()

        if self._llm_enabled:
            self.setup_tab_menu()
            main_layout.addWidget(self.tab_menu_widget)
            main_layout.addSpacing(theme.config.spacing.medium)

            self.stacked_widget = QStackedWidget()
            main_layout.addWidget(self.stacked_widget, stretch=1)

            self.stacked_widget.addWidget(self.prompts_sub_view)
            self.stacked_widget.addWidget(self.aliases_sub_view)
            self.select_tab("Prompts")
        else:
            self.tab_menu_widget = None
            self.stacked_widget = None
            main_layout.addWidget(self.aliases_sub_view, stretch=1)

    def setup_tab_menu(self) -> None:
        wrapper = TransparentBox(layout="horizontal", spacing=0)
        wrapper_layout = wrapper.layout()
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        wrapper_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        outer_container = BaseContainer(
            layout="horizontal",
            bg_color="transparent",
            border_color=theme.config.shapes.medium,
            border_radius=20,
        )
        outer_container.setMaximumWidth(500)

        outer_layout = outer_container.layout()
        outer_layout.setContentsMargins(8, 4, 8, 4)
        outer_layout.setSpacing(theme.config.spacing.medium)
        outer_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        for tab_name in ["Prompts", "Aliases"]:
            btn = TabButton(tab_name)
            btn.clicked.connect(lambda checked, name=tab_name: self.select_tab(name))
            self.tab_buttons[tab_name] = btn
            outer_layout.addWidget(btn)

        wrapper_layout.addStretch()
        wrapper_layout.addWidget(outer_container)
        wrapper_layout.addStretch()

        self.tab_menu_widget = wrapper

    def select_tab(self, tab_name: str) -> None:
        for name, btn in self.tab_buttons.items():
            btn.set_selected(name == tab_name)

        if self.stacked_widget is None:
            return

        if tab_name == "Prompts" and self.prompts_sub_view is not None:
            self.stacked_widget.setCurrentWidget(self.prompts_sub_view)
        elif tab_name == "Aliases":
            self.stacked_widget.setCurrentWidget(self.aliases_sub_view)

        self.logger.debug(f"Selected tab: {tab_name}")

    def show_main_error(self, message: str) -> None:
        QMessageBox.critical(self, "Error", message)
