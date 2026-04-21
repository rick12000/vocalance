from collections import defaultdict
from typing import Dict, List, Optional

from PySide6.QtWidgets import QDialog, QMessageBox, QVBoxLayout, QWidget

from vocalance.app.config.command_types import AutomationCommand
from vocalance.app.ui.application.base_view import QtBaseView
from vocalance.app.ui.components.buttons import ChangeButton, DangerButton, PrimaryButton
from vocalance.app.ui.components.dialogs import CommandEditDialog
from vocalance.app.ui.components.inputs import TextInput
from vocalance.app.ui.components.labels import BodyLabel, SmallLabel
from vocalance.app.ui.components.layouts import (
    CollapsibleSection,
    ListItem,
    ScrollableContainer,
    TransparentWidget,
    TwoColumnLayout,
)
from vocalance.app.ui.qt_theme import theme


class QtCommandsView(QtBaseView):
    GROUP_ORDER = ["Basic", "Window Navigation", "Editing", "General IDE", "Cursor IDE", "VSCode IDE", "Other", "Custom"]

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.commands_list: List[AutomationCommand] = []

        self.setup_ui()
        self.logger.debug("QtCommandsView initialized")

    def set_controller(self, controller) -> None:
        self.controller = controller

        self.controller.commands_loaded.connect(self.on_commands_loaded)
        self.controller.validation_error.connect(self.on_validation_error)
        self.controller.operation_error.connect(self.on_error)

        self.logger.info("Loading commands from controller")
        self.controller.on_view_ready()

    def setup_ui(self) -> None:
        self.layout = TwoColumnLayout("Add Command", "Manage Commands", self)
        self.main_layout.addWidget(self.layout, stretch=1)

        self.setup_add_command_form()
        self.setup_commands_list_panel()

    def setup_add_command_form(self) -> None:
        content = self.layout.left_content

        command_phrase_label = SmallLabel("Command Phrase:")
        content.add(command_phrase_label)
        self.command_phrase_entry = TextInput(placeholder="Enter command phrase...")
        content.add(self.command_phrase_entry)

        hotkey_label = SmallLabel("Hotkey:")
        content.add(hotkey_label)
        self.hotkey_entry = TextInput(placeholder="e.g. ctrl+alt+7")
        content.add(self.hotkey_entry)

        self.add_btn = PrimaryButton(text="Add")
        self.add_btn.clicked.connect(self.on_add_command_clicked)
        content.add(self.add_btn)

        content.add_stretch()

    def setup_commands_list_panel(self) -> None:
        content = self.layout.right_content

        self.commands_list_widget = TransparentWidget()
        self.commands_list_layout = QVBoxLayout(self.commands_list_widget)
        self.commands_list_layout.setSpacing(theme.config.container.list_item_spacing)
        self.commands_list_layout.setContentsMargins(0, 0, theme.config.spacing.small, 0)

        scroll_area = ScrollableContainer()
        scroll_area.content_layout.addWidget(self.commands_list_widget)
        content.add(scroll_area, stretch=1)

        self.reset_btn = DangerButton(text="Reset")
        self.reset_btn.clicked.connect(self.on_reset_clicked)
        content.add(self.reset_btn)

    def on_commands_loaded(self, commands: List[AutomationCommand]) -> None:
        self.commands_list = commands
        self.display_commands(commands)
        self.logger.info("Commands loaded: %s total", len(commands))

    def on_validation_error(self, error_msg: str, command_phrase: str) -> None:
        self.show_error(f"Validation error: {error_msg}")

    def on_error(self, error_msg: str) -> None:
        self.show_error(error_msg)

    def display_commands(self, commands: List[AutomationCommand]) -> None:
        while self.commands_list_layout.count() > 0:
            item = self.commands_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not commands:
            empty_label = BodyLabel("No commands available.", align="center", color=theme.config.text.medium)
            self.commands_list_layout.addWidget(empty_label)
        else:
            grouped_commands = self.group_commands(commands)
            sorted_groups = self.sort_groups(grouped_commands)

            for group_index, (group_name, group_commands) in enumerate(sorted_groups):
                is_first = group_index == 0
                collapsible_section = CollapsibleSection(group_name, is_first=is_first, start_expanded=False)

                sorted_commands = sorted(group_commands, key=lambda cmd: cmd.command_key.lower())
                for command in sorted_commands:
                    command_item = self.create_command_item_widget(command)
                    collapsible_section.add_item(command_item)

                self.commands_list_layout.addWidget(collapsible_section)

        self.commands_list_layout.addStretch()

    def group_commands(self, commands: List[AutomationCommand]) -> Dict[str, List[AutomationCommand]]:
        grouped = defaultdict(list)
        for command in commands:
            group = getattr(command, "functional_group", "Other")
            grouped[group].append(command)
        return dict(grouped)

    def sort_groups(self, grouped_commands: Dict[str, List[AutomationCommand]]) -> List[tuple]:
        sorted_groups = []

        for group_name in self.GROUP_ORDER:
            if group_name in grouped_commands:
                sorted_groups.append((group_name, grouped_commands[group_name]))

        for group_name in sorted(grouped_commands.keys()):
            if group_name not in self.GROUP_ORDER:
                sorted_groups.append((group_name, grouped_commands[group_name]))

        return sorted_groups

    def create_command_item_widget(self, command: AutomationCommand) -> QWidget:
        item = ListItem()

        phrase_label = SmallLabel(command.command_key, color=theme.config.text.medium)
        item.add(phrase_label, stretch=1)

        change_btn = ChangeButton(command=lambda checked, c=command: self.on_change_command(c))
        item.add(change_btn)

        return item

    def on_change_command(self, command: AutomationCommand) -> None:
        if not self.controller:
            self.show_error("Controller not initialized.")
            return

        dialog = CommandEditDialog(command, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            action, new_phrase = dialog.get_result()

            if action == "save" and new_phrase:
                self.controller.handle_change_command_phrase(command, new_phrase)
                self.logger.info(f"Changed command phrase: {command.command_key} -> {new_phrase}")
            elif action == "delete":
                self.controller.handle_delete_command(command)
                self.logger.info(f"Deleted command: {command.command_key}")

    def on_add_command_clicked(self) -> None:
        command_phrase = self.command_phrase_entry.text().strip()
        hotkey_value = self.hotkey_entry.text().strip()

        if not command_phrase:
            QMessageBox.warning(self, "Invalid Input", "Please enter a command phrase.")
            return

        if not hotkey_value:
            QMessageBox.warning(self, "Invalid Input", "Please enter a hotkey value.")
            return

        if not self.controller:
            QMessageBox.critical(self, "Error", "Controller not initialized.")
            return

        self.controller.handle_add_command(command_phrase, hotkey_value)
        self.command_phrase_entry.clear()
        self.hotkey_entry.clear()

    def on_reset_clicked(self) -> None:
        reply = QMessageBox.question(
            self,
            "Reset",
            "Are you sure you want to reset all commands to defaults? This will remove all custom commands.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            if self.controller:
                self.controller.handle_reset_to_defaults()

    def show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Error", message)
