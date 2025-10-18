"""
Form builder utility to simplify and standardize widget creation across views.
"""

from typing import Any, Dict, List, Optional, Tuple

import customtkinter as ctk

from iris.app.ui.views.components.themed_components import DangerButton, PrimaryButton, ThemedEntry, ThemedLabel, ThemedTextbox
from iris.app.ui.views.components.view_config import view_config


class FormBuilder:
    """Utility class for building common form patterns with minimal code"""

    def __init__(self):
        """Initialize FormBuilder with row counter starting at 0"""
        self.row = 0

    def _ensure_row_configured(self, parent: ctk.CTkFrame, row: int) -> None:
        """Ensure a specific row is configured in the parent's grid"""
        parent.grid_rowconfigure(row, weight=0)

    def setup_form_grid(self, parent: ctk.CTkFrame) -> None:
        """Configure grid weights for a standard form layout"""
        parent.grid_columnconfigure(0, weight=1)

    def create_labeled_entry(
        self,
        parent: ctk.CTkFrame,
        label_text: str,
        placeholder: str = "",
        default_value: str = "",
        row: Optional[int] = None,
        width: Optional[int] = None,
    ) -> Tuple[ThemedLabel, ThemedEntry]:
        """Create a label and entry pair with consistent styling"""
        # Use self.row if row not specified
        if row is None:
            row = self.row

        # Ensure rows are configured
        self._ensure_row_configured(parent, row)
        self._ensure_row_configured(parent, row + 1)

        label = ThemedLabel(parent, text=label_text, bold=True)
        label.grid(
            row=row,
            column=0,
            sticky="w",
            pady=(view_config.theme.spacing.medium, view_config.theme.spacing.tiny),
            padx=view_config.theme.two_box_layout.inner_content_padx,
        )

        entry_kwargs = {"placeholder_text": placeholder}
        if width:
            entry_kwargs["width"] = width

        entry = ThemedEntry(parent, **entry_kwargs)
        if default_value:
            entry.insert(0, default_value)
        entry.grid(
            row=row + 1,
            column=0,
            sticky="ew",
            pady=(0, view_config.theme.spacing.small),
            padx=view_config.theme.two_box_layout.inner_content_padx,
        )

        # Increment row counter by 2 (label + entry)
        self.row += 2

        return label, entry

    def create_labeled_textbox(
        self,
        parent: ctk.CTkFrame,
        label_text: str,
        placeholder: str = "",
        height: Optional[int] = None,
        row: Optional[int] = None,
        **kwargs,
    ) -> Tuple[ThemedLabel, ThemedTextbox]:
        """Create a label and textbox pair with consistent styling"""
        # Use self.row if row not specified
        if row is None:
            row = self.row

        # Ensure rows are configured
        self._ensure_row_configured(parent, row)
        self._ensure_row_configured(parent, row + 1)

        label = ThemedLabel(parent, text=label_text, bold=True)
        label.grid(
            row=row,
            column=0,
            sticky="w",
            pady=(view_config.theme.spacing.tiny, 0),
            padx=view_config.theme.two_box_layout.inner_content_padx,
        )

        textbox_height = height or view_config.theme.dimensions.textbox_height_small
        textbox = ThemedTextbox(
            parent,
            height=textbox_height,
            fg_color=view_config.theme.shape_colors.darkest,
            border_width=view_config.theme.entry_field_styling.border_width,
            border_color=view_config.theme.entry_field_styling.border_color,
            **kwargs,
        )
        textbox.grid(
            row=row + 1,
            column=0,
            sticky="ew",
            pady=(view_config.theme.spacing.tiny, view_config.theme.spacing.small),
            padx=view_config.theme.two_box_layout.inner_content_padx,
        )

        if placeholder:
            textbox.insert("1.0", placeholder)

        # Increment row counter by 2 (label + textbox)
        self.row += 2

        return label, textbox

    def create_button_row(
        self,
        parent: ctk.CTkFrame,
        buttons: List[Dict[str, Any]],
        row: Optional[int] = None,
        extra_pady: Optional[Tuple[int, int]] = None,
        extra_padx: Optional[int] = None,
    ) -> List[ctk.CTkButton]:
        """Create a row of buttons with consistent spacing"""
        # Use self.row if row not specified
        if row is None:
            row = self.row

        # Ensure row is configured
        self._ensure_row_configured(parent, row)

        pady = extra_pady if extra_pady is not None else view_config.theme.spacing.small
        # Use inner_content_padx if not explicitly overridden
        padx = extra_padx if extra_padx is not None else view_config.theme.two_box_layout.inner_content_padx

        button_frame = ctk.CTkFrame(parent, fg_color="transparent")
        button_frame.grid(row=row, column=0, sticky="ew", pady=pady, padx=padx)

        created_buttons = []
        for i, btn_config in enumerate(buttons):
            btn_type = btn_config.get("type", "primary")
            ButtonClass = PrimaryButton if btn_type == "primary" else DangerButton

            button = ButtonClass(button_frame, text=btn_config["text"], command=btn_config["command"])
            button.grid(
                row=0,
                column=i,
                sticky="ew",
                padx=(
                    0 if i == 0 else view_config.theme.spacing.tiny,
                    view_config.theme.spacing.tiny if i < len(buttons) - 1 else 0,
                ),
            )

            button_frame.grid_columnconfigure(i, weight=1)
            created_buttons.append(button)

        # Increment row counter by 1 (single row of buttons)
        self.row += 1

        return created_buttons

    def show_temporary_message(
        self,
        parent: ctk.CTkFrame,
        message: str,
        row: Optional[int] = None,
        duration_ms: Optional[int] = None,
        is_error: bool = True,
    ) -> None:
        """Show a temporary message that auto-dismisses"""
        # Use self.row if row not specified
        if row is None:
            row = self.row

        # Ensure row is configured
        self._ensure_row_configured(parent, row)

        if duration_ms is None:
            duration_ms = view_config.timings.error_message_display_ms

        color = view_config.theme.text_colors.darkest if is_error else view_config.theme.accent_colors.success_text
        prefix = "Warning: " if is_error else "Success: "

        msg_label = ThemedLabel(parent, text=f"{prefix}{message}", color=color)
        msg_label.grid(
            row=row,
            column=0,
            sticky="ew",
            padx=view_config.theme.two_box_layout.inner_content_padx,
            pady=view_config.theme.spacing.tiny,
        )

        def remove_message():
            try:
                if msg_label.winfo_exists():
                    msg_label.destroy()
            except Exception:
                pass

        parent.after(duration_ms, remove_message)

        # Increment row counter by 1 (temporary message row)
        self.row += 1

    def create_example_textbox_with_placeholder(
        self, parent: ctk.CTkFrame, example_text: str, row: Optional[int] = None
    ) -> ThemedTextbox:
        """Create a textbox with example text that disappears on click"""
        # Use self.row if row not specified
        if row is None:
            row = self.row

        # Ensure row is configured
        self._ensure_row_configured(parent, row)

        textbox = ThemedTextbox(
            parent,
            height=view_config.theme.dimensions.textbox_height_small,
            fg_color=view_config.theme.shape_colors.dark,
            border_width=view_config.theme.entry_field_styling.border_width,
            border_color=view_config.theme.entry_field_styling.border_color,
        )
        textbox.grid(
            row=row,
            column=0,
            sticky="nsew",
            pady=(view_config.theme.spacing.tiny, view_config.theme.spacing.small),
            padx=view_config.theme.two_box_layout.inner_content_padx,
        )

        # Insert example text with prefix
        full_example = f"{view_config.form_defaults.example_prefix}{example_text}"
        textbox.insert("1.0", full_example)

        # Make placeholder disappear on click
        def on_textbox_click(event):
            current_text = textbox.get("1.0", "end-1c")
            if current_text.startswith(view_config.form_defaults.example_prefix):
                textbox.delete("1.0", "end")

        textbox.bind("<Button-1>", on_textbox_click)

        # Increment row counter by 1 (textbox row)
        self.row += 1

        return textbox
