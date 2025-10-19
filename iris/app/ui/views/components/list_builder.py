from enum import Enum
from typing import Any, Callable, List, Optional, Tuple

import customtkinter as ctk

from iris.app.ui.views.components.themed_components import (
    BorderlessFrame,
    DangerButton,
    PrimaryButton,
    ThemedLabel,
    ThemedScrollableFrame,
)
from iris.app.ui.views.components.view_config import view_config


class ColumnType(Enum):
    """Enumeration of supported column types"""

    LABEL = "label"
    BUTTON = "button"
    CUSTOM = "custom"


class ButtonType(Enum):
    """Enumeration of button types"""

    PRIMARY = "primary"
    DANGER = "danger"


class ListItemColumn:
    """
    Configuration for a single column in a list item.

    This class uses a declarative approach where all column configurations
    are created through static factory methods, ensuring consistency.
    """

    def __init__(
        self,
        column_type: ColumnType,
        weight: int = 0,
        text: str = "",
        color: Optional[str] = None,
        anchor: str = "w",
        command: Optional[Callable] = None,
        compact: bool = True,
        button_type: ButtonType = ButtonType.PRIMARY,
        widget_factory: Optional[Callable[[ctk.CTkFrame], ctk.CTkBaseClass]] = None,
    ):
        self.column_type = column_type
        self.weight = weight
        self.text = text
        self.color = color or view_config.theme.text_colors.light
        self.anchor = anchor
        self.command = command
        self.compact = compact
        self.button_type = button_type
        self.widget_factory = widget_factory

    @staticmethod
    def label(text: str, weight: int = 1, color: Optional[str] = None, anchor: str = "w") -> "ListItemColumn":
        """Create a label column configuration"""
        return ListItemColumn(
            column_type=ColumnType.LABEL,
            weight=weight,
            text=text,
            color=color or view_config.theme.text_colors.light,
            anchor=anchor,
        )

    @staticmethod
    def button(
        text: str, command: Callable, button_type: ButtonType = ButtonType.PRIMARY, compact: bool = True
    ) -> "ListItemColumn":
        """Create a button column configuration"""
        return ListItemColumn(
            column_type=ColumnType.BUTTON,
            weight=0,
            text=text,
            command=command,
            button_type=button_type,
            compact=compact,
        )

    @staticmethod
    def widget(widget_factory: Callable[[ctk.CTkFrame], ctk.CTkBaseClass], weight: int = 0) -> "ListItemColumn":
        """
        Create a custom widget column configuration.

        Args:
            widget_factory: Callable that takes parent frame and returns configured widget
            weight: Grid weight for the column
        """
        return ListItemColumn(
            column_type=ColumnType.CUSTOM,
            weight=weight,
            widget_factory=widget_factory,
        )


class ListBuilder:
    """Utility class for building scrollable lists with consistent styling"""

    @staticmethod
    def create_scrollable_list_container(
        parent: ctk.CTkFrame,
        row: int = 0,
        column: int = 0,
        padx: Optional[int] = None,
        pady: Optional[Tuple[int, int]] = None,
    ) -> ThemedScrollableFrame:
        """
        Create a themed scrollable frame for list content with standard styling.

        Args:
            parent: Parent container (usually the content area from TwoColumnTabLayout)
            row: Grid row position
            column: Grid column position
            padx: Horizontal padding (defaults to asymmetric - 0 left, scrollbar_right_padding right)
            pady: Vertical padding tuple (defaults to standard list padding)

        Returns:
            ThemedScrollableFrame configured for list content
        """
        if padx is None:
            # Asymmetric padding: no left padding, right padding for scrollbar clearance
            padx = (0, view_config.theme.list_layout.scrollbar_right_padding)

        if pady is None:
            pady = (0, view_config.theme.spacing.small)

        scroll_frame = ThemedScrollableFrame(parent)
        scroll_frame.grid(
            row=row,
            column=column,
            sticky="nsew",
            padx=padx,
            pady=pady,
        )

        scroll_frame.grid_columnconfigure(0, weight=1)

        return scroll_frame

    @staticmethod
    def create_scrollable_list_container_with_padx(
        parent: ctk.CTkFrame,
        left_padx: int = 0,
        row: int = 0,
        column: int = 0,
        pady: Optional[Tuple[int, int]] = None,
    ) -> ThemedScrollableFrame:
        """
        Create a themed scrollable frame with custom left padding.
        Used for special views like settings that have different left padding.
        Always applies scrollbar_right_padding to maintain consistent scrollbar spacing.

        Args:
            parent: Parent container
            left_padx: Left padding (default 0)
            row: Grid row position
            column: Grid column position
            pady: Vertical padding tuple

        Returns:
            ThemedScrollableFrame configured for list content
        """
        if pady is None:
            pady = (0, view_config.theme.spacing.small)

        scroll_frame = ThemedScrollableFrame(parent)
        scroll_frame.grid(
            row=row,
            column=column,
            sticky="nsew",
            padx=(left_padx, view_config.theme.list_layout.scrollbar_right_padding),
            pady=pady,
        )

        scroll_frame.grid_columnconfigure(0, weight=1)

        return scroll_frame

    @staticmethod
    def display_items(
        container: ThemedScrollableFrame,
        items: List[Any],
        create_item_callback: Callable[[Any, int], None],
        empty_message: Optional[str] = None,
    ) -> None:
        """
        Display items in a scrollable list with optional empty state.

        Args:
            container: The scrollable frame container
            items: List of items to display
            create_item_callback: Callback function(item, row_index) to create each item
            empty_message: Optional message to display when list is empty
        """
        for widget in container.winfo_children():
            widget.destroy()

        container.grid_columnconfigure(0, weight=1)

        if not items and empty_message:
            ListBuilder._create_empty_state(container, empty_message)
        else:
            for i, item in enumerate(items):
                create_item_callback(item, i)

    @staticmethod
    def _create_empty_state(container: ThemedScrollableFrame, message: str) -> None:
        """Create an empty state message in the list"""
        empty_frame = BorderlessFrame(container)
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
            text=message,
            anchor="center",
            color=view_config.theme.text_colors.medium,
            size=view_config.theme.font_sizes.medium,
        ).grid(
            row=0,
            column=0,
            sticky="ew",
            padx=view_config.theme.spacing.medium,
            pady=view_config.theme.spacing.large,
        )

    @staticmethod
    def create_list_item(
        container: ThemedScrollableFrame,
        row_index: int,
        columns: List[ListItemColumn],
        padx: Optional[int] = None,
        pady: Optional[int] = None,
    ) -> BorderlessFrame:
        """
        Create a list item with configurable columns.

        Args:
            container: Parent scrollable frame
            row_index: Row index in the list
            columns: List of ListItemColumn configurations
            padx: Horizontal padding (defaults to item_padx - small padding for nested items)
            pady: Vertical padding (defaults to item_vertical_spacing)

        Returns:
            The created item frame
        """
        if padx is None:
            # Use asymmetric padding: small left, extra right for scrollbar spacing
            padx = (view_config.theme.list_layout.item_padx, view_config.theme.list_layout.item_padx_right)

        if pady is None:
            pady = view_config.theme.list_layout.item_vertical_spacing

        item_frame = BorderlessFrame(container)
        item_frame.grid(
            row=row_index,
            column=0,
            sticky="ew",
            padx=padx,
            pady=pady,
        )

        for col_idx, column_config in enumerate(columns):
            item_frame.grid_columnconfigure(col_idx, weight=column_config.weight)

            ListBuilder._create_column_widget(item_frame, column_config, col_idx)

        return item_frame

    @staticmethod
    def _create_column_widget(parent_frame: BorderlessFrame, column_config: ListItemColumn, col_idx: int) -> None:
        """Create a widget for a specific column based on configuration"""
        # Use spacing.tiny for internal column spacing (between columns)
        padx = view_config.theme.spacing.tiny

        if column_config.column_type == ColumnType.LABEL:
            widget = ThemedLabel(
                parent_frame,
                text=column_config.text,
                anchor=column_config.anchor,
                color=column_config.color,
            )
            widget.grid(
                row=0,
                column=col_idx,
                sticky="ew" if column_config.weight > 0 else column_config.anchor,
                padx=padx,
            )

        elif column_config.column_type == ColumnType.BUTTON:
            button_class = PrimaryButton if column_config.button_type == ButtonType.PRIMARY else DangerButton
            widget = button_class(
                parent_frame,
                text=column_config.text,
                compact=column_config.compact,
                command=column_config.command,
            )
            widget.grid(
                row=0,
                column=col_idx,
                padx=padx,
            )

        elif column_config.column_type == ColumnType.CUSTOM and column_config.widget_factory:
            widget = column_config.widget_factory(parent_frame)
            widget.grid(
                row=0,
                column=col_idx,
                padx=padx,
            )

    @staticmethod
    def create_group_header(
        container: ThemedScrollableFrame,
        row_index: int,
        group_name: str,
    ) -> BorderlessFrame:
        """
        Create a group header with a title.

        Args:
            container: Parent scrollable frame
            row_index: Row index in the list
            group_name: Name of the group

        Returns:
            The created header frame
        """
        header_frame = BorderlessFrame(container)
        header_frame.grid(
            row=row_index,
            column=0,
            sticky="ew",
            padx=(view_config.theme.list_layout.item_padx, view_config.theme.list_layout.item_padx_right),
            pady=(view_config.theme.spacing.medium, view_config.theme.spacing.tiny),
        )

        header_frame.grid_columnconfigure(0, weight=1)

        ThemedLabel(
            header_frame,
            text=group_name,
            anchor="w",
            color=view_config.theme.text_colors.lightest,
            size=view_config.theme.font_sizes.medium,
            bold=True,
        ).grid(
            row=0,
            column=0,
            sticky="w",
            padx=0,
        )

        return header_frame

    @staticmethod
    def create_divider(
        container: ThemedScrollableFrame,
        row_index: int,
    ) -> ctk.CTkFrame:
        """
        Create a horizontal divider line using canvas for reliable pixel-perfect rendering.

        Args:
            container: Parent scrollable frame
            row_index: Row index in the list

        Returns:
            The created divider frame
        """
        import tkinter as tk

        divider_frame = BorderlessFrame(container)
        divider_frame.grid(
            row=row_index,
            column=0,
            sticky="ew",
            padx=(view_config.theme.list_layout.item_padx, view_config.theme.list_layout.item_padx_right),
            pady=(view_config.theme.spacing.tiny, view_config.theme.spacing.tiny),
        )

        divider_frame.grid_columnconfigure(0, weight=1)
        divider_frame.grid_rowconfigure(0, weight=0, minsize=1)

        # Use Canvas instead of CTkFrame for reliable 1px line rendering across all DPI settings
        divider_canvas = tk.Canvas(
            divider_frame,
            height=1,
            bg=view_config.theme.shape_colors.medium,
            highlightthickness=0,
            bd=0,
        )
        divider_canvas.grid(row=0, column=0, sticky="ew")

        # Draw the line on canvas - this ensures pixel-perfect rendering
        def draw_divider(event=None):
            divider_canvas.delete("divider_line")
            canvas_width = divider_canvas.winfo_width()
            if canvas_width > 1:
                divider_canvas.create_line(
                    0, 0, canvas_width, 0, fill=view_config.theme.shape_colors.medium, width=1, tags="divider_line"
                )

        # Bind to configure event to redraw when resized
        divider_canvas.bind("<Configure>", draw_divider)

        # Initial draw after canvas is rendered
        divider_canvas.after(10, draw_divider)

        return divider_frame
