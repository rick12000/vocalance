from typing import List

from iris.app.events.mark_events import MarkData
from iris.app.ui.controls.marks_control import MarksController
from iris.app.ui.views.components.base_view import ViewHelper
from iris.app.ui.views.components.themed_components import (
    BorderlessFrame,
    BorderlessListItemFrame,
    DangerButton,
    InstructionTile,
    PrimaryButton,
    ThemedLabel,
    ThemedScrollableFrame,
    TransparentFrame,
    TwoColumnTabLayout,
)
from iris.app.ui.views.components.view_config import view_config


class MarksView(ViewHelper):
    """Simplified marks view using base components"""

    def __init__(self, parent, controller: MarksController):
        super().__init__(parent, controller)
        self._setup_ui()
        self.controller.refresh_marks()

    def _setup_ui(self) -> None:
        """Setup the main UI layout"""
        self.setup_main_layout()

        self.layout = TwoColumnTabLayout(self, "Instructions", "Manage Marks")
        self.layout.grid(row=0, column=0, sticky="nsew")

        self._setup_instructions_panel()
        self._setup_marks_panel()

    def _setup_instructions_panel(self) -> None:
        """Setup voice commands instructions panel"""
        container = self.layout.left_content
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # Container for tiles with proper padding
        tiles_container = TransparentFrame(container)
        tiles_container.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=view_config.theme.two_box_layout.box_content_padding,
            pady=(0, view_config.theme.two_box_layout.last_element_bottom_padding),
        )

        # Configure grid for 3 rows, 1 column
        for i in range(3):
            tiles_container.grid_rowconfigure(i, weight=1)
        tiles_container.grid_columnconfigure(0, weight=1)

        # Create instruction tiles
        instructions = [
            ("Create Mark", "Say 'Mark [name]' to create a mark\nat the current cursor position"),
            ("Navigate", "Say the mark's [name] to automatically click\nat that position"),
            ("Manage Marks", "Use the right panel to visualize and delete marks,\nor say 'show marks' to see them on screen"),
        ]

        for i, (title, content) in enumerate(instructions):
            tile = InstructionTile(tiles_container, title=title, content=content)
            tile.grid(
                row=i,
                column=0,
                sticky="nsew",
                padx=view_config.theme.tile_layout.padding_between_tiles,
                pady=view_config.theme.tile_layout.padding_between_tiles,
            )

    def _setup_marks_panel(self) -> None:
        """Setup marks management panel"""
        container = self.layout.right_content
        container.grid_rowconfigure(0, weight=1)
        container.grid_rowconfigure(1, weight=0)
        container.grid_columnconfigure(0, weight=1)

        # Scrollable frame for marks list
        self.marks_scroll_frame = ThemedScrollableFrame(container)
        self.marks_scroll_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)

        # Button frame with bottom padding for rounded corners
        button_frame = TransparentFrame(container)
        button_frame.grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(view_config.theme.spacing.small, view_config.theme.two_box_layout.last_element_bottom_padding),
            padx=view_config.theme.two_box_layout.box_content_padding,
        )
        button_frame.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(1, weight=1)

        # Show overlay button
        PrimaryButton(button_frame, text=view_config.theme.button_text.show_marks, command=self._show_overlay).grid(
            row=0, column=0, padx=(0, view_config.theme.spacing.small), sticky="ew"
        )

        # Delete all marks button
        DangerButton(button_frame, text=view_config.theme.button_text.delete_all_marks, command=self._delete_all_marks).grid(
            row=0, column=1, sticky="ew"
        )

    def _show_overlay(self) -> None:
        """Show marks overlay"""
        self.controller.request_show_overlay()

    def _delete_all_marks(self) -> None:
        """Delete all marks with confirmation"""
        if self.show_delete_all_confirmation("marks"):
            self.controller.delete_all_marks()

    def display_marks(self, marks: List[MarkData]) -> None:
        """Display marks in the scrollable list"""
        self.logger.debug(f"MarksView: Received {len(marks)} marks to display.")

        # Clear existing marks safely
        if not self.safe_widget_operation(lambda: self.marks_scroll_frame.winfo_exists()):
            self.logger.warning("MarksView: marks_scroll_frame is not available for clearing.")
            return

        # Clear existing widgets
        for widget in self.marks_scroll_frame.winfo_children():
            self.safe_widget_operation(lambda w=widget: w.destroy())

        # Force update to ensure widgets are destroyed
        self.safe_widget_operation(lambda: self.marks_scroll_frame.update_idletasks())

        # Configure scroll frame grid
        self.marks_scroll_frame.grid_columnconfigure(0, weight=1)

        if not marks:
            # Show empty state message
            empty_frame = BorderlessFrame(self.marks_scroll_frame)
            empty_frame.grid(
                row=0,
                column=0,
                sticky="ew",
                padx=view_config.theme.spacing.tiny,
                pady=view_config.theme.list_layout.item_vertical_spacing,
            )

            empty_frame.grid_columnconfigure(0, weight=1)

            ThemedLabel(
                empty_frame,
                text="No available marks.\nFollow the instructions on the left panel to create a mark.",
                anchor="center",
                color=view_config.theme.text_colors.medium,
                size=view_config.theme.font_sizes.medium,
            ).grid(row=0, column=0, sticky="ew", padx=view_config.theme.spacing.medium, pady=view_config.theme.spacing.large)
        else:
            # Add each mark
            for row_idx, mark in enumerate(marks):
                try:
                    if not self.safe_widget_operation(lambda: self.marks_scroll_frame.winfo_exists()):
                        break

                    # Create mark info text
                    mark_info = mark.name
                    if mark.description:
                        mark_info += f" - {mark.description}"

                    # Create list item frame
                    mark_frame = BorderlessListItemFrame(
                        self.marks_scroll_frame,
                        item_text=mark_info,
                        button_text=view_config.theme.button_text.delete,
                        button_command=lambda m=mark.name: self._delete_mark(m),
                        button_variant="danger",
                    )
                    mark_frame.grid(
                        row=row_idx,
                        column=0,
                        sticky="ew",
                        padx=view_config.theme.spacing.tiny,
                        pady=(view_config.theme.list_layout.item_vertical_spacing, 0),
                    )

                except Exception as e:
                    self.logger.error(f"Error creating mark frame for {mark.name}: {e}")

    def _delete_mark(self, mark_name: str) -> None:
        """Delete a specific mark"""
        if self.show_delete_confirmation(f"mark '{mark_name}'"):
            self.controller.delete_mark_by_name(mark_name)

    def refresh_marks_list(self) -> None:
        """Refresh the marks list"""
        self.controller.refresh_marks()

    # Controller callback methods
    def on_marks_updated(self, marks: List[MarkData]) -> None:
        """Handle marks list updated from controller"""
        self.display_marks(marks)

    def on_status_update(self, message: str, is_error: bool = False) -> None:
        """Handle status updates from controller"""
        super().on_status_update(message, is_error)
