import logging
from collections import defaultdict
from typing import Dict, List

from vocalance.app.config.command_types import AutomationCommand
from vocalance.app.ui.views.command_dialog_view import CommandEditDialog
from vocalance.app.ui.views.components.base_view import ViewHelper
from vocalance.app.ui.views.components.form_builder import FormBuilder
from vocalance.app.ui.views.components.list_builder import ButtonType, ListBuilder, ListItemColumn
from vocalance.app.ui.views.components.themed_components import TwoColumnTabLayout
from vocalance.app.ui.views.components.view_config import view_config


class CommandsView(ViewHelper):
    """Simplified commands view using base components and form builder"""

    GROUP_ORDER = ["Basic", "Window Navigation", "Editing", "General IDE", "Cursor IDE", "VSCode IDE", "Other", "Custom"]

    def __init__(self, parent, controller, root_window, logger: logging.Logger):
        super().__init__(parent, controller, root_window)
        self.logger = logger
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Setup the main UI layout"""
        self.setup_main_layout()

        self.layout = TwoColumnTabLayout(self, "Add Command", "Manage Commands")
        self.layout.grid(row=0, column=0, sticky="nsew")

        self._setup_add_command_form()
        self._setup_commands_list_panel()

    def _setup_add_command_form(self) -> None:
        """Setup add command form using form builder"""
        container = self.layout.left_content
        form_builder = FormBuilder()
        form_builder.setup_form_grid(container)

        # Form fields
        self.phrase_label, self.command_phrase_entry = form_builder.create_labeled_entry(
            container, "Command Phrase:", "Enter command phrase..."
        )

        self.hotkey_label, self.hotkey_entry = form_builder.create_labeled_entry(container, "Hotkey:", "e.g., ctrl+alt+7")

        # Add button
        form_builder.create_button_row(
            container,
            [
                {
                    "text": view_config.theme.button_text.add_command,
                    "command": self._on_add_command_clicked,
                    "type": "primary",
                    "compact": False,
                }
            ],
        )

    def _setup_commands_list_panel(self) -> None:
        """Setup commands list panel"""
        container = self.layout.right_content
        container.grid_rowconfigure(0, weight=1)
        container.grid_rowconfigure(1, weight=0)
        container.grid_columnconfigure(0, weight=1)

        self.command_list_container = ListBuilder.create_scrollable_list_container(container, row=0, column=0)

        form_builder = FormBuilder()
        form_builder.create_button_row(
            container,
            [
                {
                    "text": view_config.theme.button_text.reset,
                    "command": self._on_reset_to_defaults_clicked,
                    "type": "danger",
                    "compact": False,
                }
            ],
            extra_pady=(0, view_config.theme.two_box_layout.last_element_bottom_padding),
            extra_padx=view_config.theme.two_box_layout.box_content_padding,
            row=1,
        )

    def _on_add_command_clicked(self) -> None:
        """Handle add command button click"""
        command_phrase = self.command_phrase_entry.get().strip()
        hotkey_value = self.hotkey_entry.get().strip()
        self.controller.handle_add_command(command_phrase, hotkey_value)

    def clear_add_command_form(self) -> None:
        """Clear the add command form"""
        self.clear_form_fields(self.command_phrase_entry, self.hotkey_entry)

    def display_commands(self, commands: List[AutomationCommand]) -> None:
        """Display commands in a scrollable list grouped by functional_group"""
        self.logger.info(f"CommandsView: display_commands called with {len(commands)} commands")

        for widget in self.command_list_container.winfo_children():
            widget.destroy()

        self.command_list_container.grid_columnconfigure(0, weight=1)

        grouped_commands = self._group_commands_by_functional_group(commands)
        sorted_groups = self._sort_groups(grouped_commands)

        row_idx = 0
        for group_idx, (group_name, group_commands) in enumerate(sorted_groups):
            ListBuilder.create_group_header(
                container=self.command_list_container,
                row_index=row_idx,
                group_name=group_name,
            )
            row_idx += 1

            ListBuilder.create_divider(container=self.command_list_container, row_index=row_idx)
            row_idx += 1

            sorted_commands = sorted(group_commands, key=lambda cmd: cmd.command_key.lower())

            for command in sorted_commands:
                self._create_command_item(command, row_idx)
                row_idx += 1

        self.logger.info(f"CommandsView: Finished displaying {len(commands)} commands in {len(sorted_groups)} groups")

    def _group_commands_by_functional_group(self, commands: List[AutomationCommand]) -> Dict[str, List[AutomationCommand]]:
        """Group commands by their functional_group attribute"""
        grouped = defaultdict(list)
        for command in commands:
            group = getattr(command, "functional_group", "Other")
            grouped[group].append(command)
        return dict(grouped)

    def _sort_groups(self, grouped_commands: Dict[str, List[AutomationCommand]]) -> List[tuple]:
        """Sort groups according to GROUP_ORDER"""
        sorted_groups = []

        for group_name in self.GROUP_ORDER:
            if group_name in grouped_commands:
                sorted_groups.append((group_name, grouped_commands[group_name]))

        for group_name in sorted(grouped_commands.keys()):
            if group_name not in self.GROUP_ORDER:
                sorted_groups.append((group_name, grouped_commands[group_name]))

        return sorted_groups

    def _create_command_item(self, command: AutomationCommand, row_idx: int) -> None:
        """Create a streamlined command list item with only trigger word and change button"""
        ListBuilder.create_list_item(
            container=self.command_list_container,
            row_index=row_idx,
            columns=[
                ListItemColumn.label(text=command.command_key, weight=1),
                ListItemColumn.button(
                    text=view_config.theme.button_text.change,
                    command=lambda c=command: self.handle_change_command(c),
                    button_type=ButtonType.PRIMARY,
                ),
            ],
        )

    def handle_change_command(self, command: AutomationCommand) -> None:
        """Handle changing a command"""
        dialog = CommandEditDialog(command, self.root_window)
        action, new_phrase = dialog.show()

        if action == "save" and new_phrase:
            self.controller.handle_change_command_phrase(command, new_phrase)
        elif action == "delete":
            self.controller.handle_delete_command(command)

    def handle_delete_command(self, command: AutomationCommand) -> None:
        """Handle deleting a command"""
        self.controller.handle_delete_command(command)

    def _on_reset_to_defaults_clicked(self) -> None:
        """Handle reset to defaults button click"""
        if self.show_confirmation(
            "Reset to Defaults", "Are you sure you want to reset all commands to defaults? This will remove all custom commands."
        ):
            self.controller.handle_reset_to_defaults()

    # Simplified callback methods - inherit most from BaseView
    def show_error_message(self, title: str, message: str) -> None:
        """Show an error message dialog"""
        self.show_error(title, message)

    def show_success_message(self, title: str, message: str) -> None:
        """Show a success message dialog"""
        self.show_info(title, message)

    def show_confirmation_dialog(self, title: str, message: str) -> bool:
        """Show a confirmation dialog and return the result"""
        return self.show_confirmation(title, message)
