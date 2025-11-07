import customtkinter as ctk

from vocalance.app.ui import ui_theme
from vocalance.app.ui.controls.settings_control import SettingsController
from vocalance.app.ui.views.components import themed_dialogs as messagebox
from vocalance.app.ui.views.components.list_builder import ListBuilder
from vocalance.app.ui.views.components.themed_components import (
    BoxTitle,
    DangerButton,
    PrimaryButton,
    ThemedEntry,
    ThemedFrame,
    ThemedLabel,
    TransparentFrame,
)


class SettingsView(ctk.CTkFrame):
    """UI view for settings tab - handles settings configuration with real-time updates"""

    def __init__(self, parent_frame, controller: SettingsController, root_window):
        super().__init__(parent_frame, fg_color=ui_theme.theme.shape_colors.darkest)

        self.controller = controller
        self.parent_frame = parent_frame
        self.root_window = root_window
        self._is_alive = True

        # Store entry widget references to manage focus
        self._entry_widgets = []

        # Create string variables for all settings
        self.llm_context_length_var = ctk.StringVar()
        self.llm_max_tokens_var = ctk.StringVar()
        self.grid_default_cells_var = ctk.StringVar()
        self.markov_enabled_var = ctk.StringVar(value="No")  # Default to disabled
        self.markov_confidence_var = ctk.StringVar()
        self.sound_confidence_var = ctk.StringVar()
        self.sound_vote_var = ctk.StringVar()
        self.dictation_silent_chunks_var = ctk.StringVar()
        self.command_silent_chunks_var = ctk.StringVar()

        self._build_tab_ui()
        self._load_current_settings()

        self.controller.set_view_callback(self)

    def _create_settings_section(self, parent, title, section_index, fields, save_command, reset_command, is_last_section=False):
        """
        Create a reusable settings section with 3-column layout and info buttons.

        Args:
            parent: Parent frame
            title: Section title
            section_index: Index of this section (for grid positioning)
            fields: List of (label, variable, description) or (label, variable, description, options) tuples
                   If 4th element (options) is provided, creates a dropdown instead of entry
            save_command: Save button command
            reset_command: Reset button command
            is_last_section: If True, no divider line is drawn below this section
        """
        base_row = section_index * 20

        header = BoxTitle(parent, text=title)
        header.grid(
            row=base_row,
            column=0,
            columnspan=3,
            padx=ui_theme.theme.two_box_layout.outer_padding_left,
            pady=(
                ui_theme.theme.spacing.medium if section_index == 0 else ui_theme.theme.spacing.large,
                ui_theme.theme.spacing.small,
            ),
            sticky="w",
        )

        for idx, field in enumerate(fields):
            current_row = base_row + 1 + idx

            # Support both 3-tuple (entry) and 4-tuple (dropdown)
            if len(field) == 4:
                label_text, variable, description, options = field
                is_dropdown = True
            else:
                label_text, variable, description = field
                is_dropdown = False

            label = ThemedLabel(parent, text=label_text, bold=True)
            label.grid(
                row=current_row,
                column=0,
                padx=(ui_theme.theme.two_box_layout.outer_padding_left, ui_theme.theme.spacing.medium),
                pady=ui_theme.theme.spacing.small,
                sticky="w",
            )

            if is_dropdown:
                # Create dropdown with consistent styling matching page theme
                dropdown = ctk.CTkComboBox(
                    parent,
                    variable=variable,
                    values=options,
                    state="readonly",
                    fg_color=ui_theme.theme.shape_colors.darkest,  # Match page background
                    border_color=ui_theme.theme.shape_colors.medium,
                    button_color=ui_theme.theme.shape_colors.medium,
                    button_hover_color=ui_theme.theme.shape_colors.lightest,
                    dropdown_fg_color=ui_theme.theme.shape_colors.darkest,  # Match page background
                    dropdown_hover_color=ui_theme.theme.shape_colors.medium,
                    text_color=ui_theme.theme.text_colors.light,
                    width=200,
                )
                dropdown.grid(
                    row=current_row,
                    column=1,
                    padx=(0, ui_theme.theme.spacing.medium),
                    pady=ui_theme.theme.spacing.small,
                    sticky="ew",
                )
            else:
                entry = ThemedEntry(parent, textvariable=variable)
                entry.grid(
                    row=current_row,
                    column=1,
                    padx=(0, ui_theme.theme.spacing.medium),
                    pady=ui_theme.theme.spacing.small,
                    sticky="ew",
                )
                # Store entry widget reference for focus management
                self._entry_widgets.append(entry)

            info_button = PrimaryButton(parent, text="Info", command=lambda desc=description: self._show_info_dialog(desc))
            info_button.grid(
                row=current_row,
                column=2,
                padx=(0, ui_theme.theme.two_box_layout.outer_padding_right),
                pady=ui_theme.theme.spacing.small,
                sticky="e",
            )

        buttons_row = base_row + len(fields) + 1
        buttons_frame = TransparentFrame(parent)
        buttons_frame.grid(
            row=buttons_row,
            column=0,
            columnspan=3,
            pady=(ui_theme.theme.spacing.medium, ui_theme.theme.spacing.medium),
            sticky="w",
        )

        buttons_frame.grid_columnconfigure(0, weight=0)
        buttons_frame.grid_columnconfigure(1, weight=0)

        PrimaryButton(buttons_frame, text="Save", command=save_command).grid(
            row=0, column=0, padx=(ui_theme.theme.two_box_layout.outer_padding_left, ui_theme.theme.spacing.small)
        )
        DangerButton(buttons_frame, text="Reset", command=reset_command).grid(row=0, column=1)

        if not is_last_section:
            divider_row = buttons_row + 1
            divider = ThemedFrame(parent, height=1, fg_color=ui_theme.theme.shape_colors.lightest)
            divider.grid(
                row=divider_row,
                column=0,
                columnspan=3,
                padx=(ui_theme.theme.two_box_layout.outer_padding_left, ui_theme.theme.two_box_layout.outer_padding_right),
                pady=(ui_theme.theme.spacing.medium, 0),
                sticky="ew",
            )

    def _show_info_dialog(self, description: str):
        """Show info dialog with setting description"""
        messagebox.showinfo(description, parent=self.root_window)

    def _build_tab_ui(self):
        """Build the settings tab UI with 3-column layout"""
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Outer container with border and padding from parent
        container_frame = ctk.CTkFrame(
            self,
            fg_color=ui_theme.theme.shape_colors.dark,
            corner_radius=ui_theme.theme.two_box_layout.box_corner_radius,
            border_width=1,
            border_color=ui_theme.theme.shape_colors.medium,
        )
        container_frame.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(ui_theme.theme.two_box_layout.outer_padding_left, ui_theme.theme.two_box_layout.outer_padding_right),
            pady=(ui_theme.theme.two_box_layout.outer_padding_top, ui_theme.theme.two_box_layout.outer_padding_bottom),
        )

        container_frame.grid_rowconfigure(0, weight=1)
        container_frame.grid_columnconfigure(0, weight=1)

        # Inner content frame to provide proper spacing from border
        content_frame = ctk.CTkFrame(
            container_frame,
            fg_color="transparent",  # Transparent to show container background
            corner_radius=0,  # No corner radius for inner frame
            border_width=0,  # No border for inner frame
        )
        content_frame.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=ui_theme.theme.spacing.medium,  # Consistent padding from border
            pady=ui_theme.theme.spacing.medium,
        )

        content_frame.grid_rowconfigure(0, weight=1)
        content_frame.grid_columnconfigure(0, weight=1)

        # Scrollable content inside the content frame
        scrollable_frame = ListBuilder.create_scrollable_list_container(
            content_frame,
            padx=0,  # No additional padding, handled by content_frame
            pady=0,  # No additional padding, handled by content_frame
        )

        scrollable_frame.grid_columnconfigure(0, weight=0)
        scrollable_frame.grid_columnconfigure(1, weight=1)
        scrollable_frame.grid_columnconfigure(2, weight=0)

        self._create_settings_section(
            scrollable_frame,
            "LLM Model Settings",
            0,
            [
                (
                    "Max Context Tokens:",
                    self.llm_context_length_var,
                    "Maximum words (1 token ≈ 1 word) the LLM can retain in memory at any given time. Increase if you plan to dictate longer texts.",
                ),
                (
                    "Max Output Tokens:",
                    self.llm_max_tokens_var,
                    "Maximum words the LLM can output (1 token ≈ 1 word). Increase if you want the model to be able to output more words.",
                ),
            ],
            self._save_llm_settings,
            self._reset_llm_to_defaults,
        )

        self._create_settings_section(
            scrollable_frame,
            "Grid Settings",
            1,
            [
                (
                    "Default Cell Count:",
                    self.grid_default_cells_var,
                    "The number of rectangles that will be displayed in the grid UI overlay.",
                )
            ],
            self._save_grid_settings,
            self._reset_grid_to_defaults,
        )

        self._create_settings_section(
            scrollable_frame,
            "Markov Chain Settings",
            2,
            [
                (
                    "Enable:",
                    self.markov_enabled_var,
                    "Enable or disable Markov chain command prediction. When enabled, commands are predicted and executed before speech recognition completes for ultra-low latency.",
                    ["No", "Yes"],
                ),
                (
                    "Prediction Confidence:",
                    self.markov_confidence_var,
                    "Threshold for Markov model to execute predicted commands. At 0.95: 95% correct. At 1.0: almost never wrong, but rarely triggers.",
                ),
            ],
            self._save_markov_settings,
            self._reset_markov_to_defaults,
        )

        self._create_settings_section(
            scrollable_frame,
            "Sound Recognizer Settings",
            3,
            [
                (
                    "Confidence Threshold:",
                    self.sound_confidence_var,
                    "Minimum cosine similarity required for sound recognition. Increase if sounds get misrecognized. Decrease if sounds aren't detected.",
                ),
                (
                    "Vote Threshold:",
                    self.sound_vote_var,
                    "Agreement level among nearest neighbor sound labels. Increase if sounds get misrecognized. Decrease if sounds aren't detected.",
                ),
            ],
            self._save_sound_settings,
            self._reset_sound_to_defaults,
        )

        self._create_settings_section(
            scrollable_frame,
            "Voice Settings",
            4,
            [
                (
                    "Max Silent Dictation Chunks:",
                    self.dictation_silent_chunks_var,
                    "Number of consecutive silent audio chunks before a dictation segment is transcribed. Increase this if you want to talk for longer before seeing the transcription (helps with formatting and punctuation). 1 chunk = 50ms",
                ),
                (
                    "Max Silent Command Chunks:",
                    self.command_silent_chunks_var,
                    "Number of consecutive silent audio chunks before a command segment is processed. Increase this if commands are being cut off too early. 1 chunk = 50ms",
                ),
            ],
            self._save_voice_settings,
            self._reset_voice_to_defaults,
            is_last_section=True,
        )

    def _force_entry_updates(self):
        """Force all entry widgets to update their displayed values from StringVars"""
        try:
            # Remove focus from all entries by focusing on root
            self.root_window.focus_set()
            self.root_window.update_idletasks()

            # Force each entry to update by deleting and reinserting from StringVar
            for entry in self._entry_widgets:
                try:
                    if entry.winfo_exists():
                        var = entry.cget("textvariable")
                        if var:
                            new_value = var.get()
                            current_value = entry.get()

                            if new_value != current_value:
                                entry.delete(0, "end")
                                entry.insert(0, new_value)
                except Exception as e:
                    self.controller.logger.warning(f"Could not update entry widget: {e}")

            self.update_idletasks()
        except Exception as e:
            self.controller.logger.error(f"Could not force entry updates: {e}", exc_info=True)

    def on_settings_updated(self):
        """Handle settings updated event from controller"""
        if not self._is_alive:
            return
        self._load_current_settings()
        self._force_entry_updates()

    def on_validation_error(self, title: str, message: str):
        """Handle validation errors from controller"""
        messagebox.showerror(message, parent=self.root_window)

    def on_save_success(self, message: str):
        """Handle successful save from controller"""
        self._load_current_settings()
        self._force_entry_updates()
        messagebox.showinfo(message, parent=self.root_window)

    def on_save_error(self, message: str):
        """Handle save errors from controller"""
        messagebox.showerror(message, parent=self.root_window)

    def on_reset_complete(self):
        """Handle reset completion from controller"""
        self._load_current_settings()
        self._force_entry_updates()
        messagebox.showinfo("Settings have been reset to defaults", parent=self.root_window)

    def _load_current_settings(self):
        """Load current settings from controller and update UI"""
        try:
            settings = self.controller.load_current_settings()

            if settings:
                llm_settings = settings.get("llm", {})
                self.llm_context_length_var.set(str(llm_settings.get("context_length", 2048)))
                self.llm_max_tokens_var.set(str(llm_settings.get("max_tokens", 512)))

                grid_settings = settings.get("grid", {})
                self.grid_default_cells_var.set(str(grid_settings.get("default_rect_count", 500)))

                markov_settings = settings.get("markov_predictor", {})
                markov_enabled = markov_settings.get("enabled", False)
                self.markov_enabled_var.set("Yes" if markov_enabled else "No")
                self.markov_confidence_var.set(str(markov_settings.get("confidence_threshold", 0.85)))

                sound_settings = settings.get("sound_recognizer", {})
                self.sound_confidence_var.set(str(sound_settings.get("confidence_threshold", 0.15)))
                self.sound_vote_var.set(str(sound_settings.get("vote_threshold", 0.35)))

                vad_settings = settings.get("vad", {})
                self.dictation_silent_chunks_var.set(str(vad_settings.get("dictation_silent_chunks_for_end", 16)))
                self.command_silent_chunks_var.set(str(vad_settings.get("command_silent_chunks_for_end", 4)))
            else:
                # Set error state if settings could not be loaded
                for var in [
                    self.llm_context_length_var,
                    self.llm_max_tokens_var,
                    self.grid_default_cells_var,
                    self.markov_confidence_var,
                    self.sound_confidence_var,
                    self.sound_vote_var,
                    self.dictation_silent_chunks_var,
                    self.command_silent_chunks_var,
                ]:
                    if isinstance(var, ctk.StringVar):
                        var.set("Error")

        except Exception as e:
            self.controller.logger.error(f"Error loading settings into UI: {e}")

    def _save_llm_settings(self):
        """Save LLM settings through controller"""
        self.controller.save_llm_settings(self.llm_context_length_var.get(), self.llm_max_tokens_var.get())

    def _reset_llm_to_defaults(self):
        """Reset LLM settings to defaults through controller"""
        if messagebox.askyesno("Are you sure you want to reset LLM settings to defaults?", parent=self.root_window):
            self.controller.reset_llm_to_defaults()

    def _save_grid_settings(self):
        """Save Grid settings through controller"""
        self.controller.save_grid_settings(self.grid_default_cells_var.get())

    def _reset_grid_to_defaults(self):
        """Reset Grid settings to defaults through controller"""
        if messagebox.askyesno("Are you sure you want to reset Grid settings to defaults?", parent=self.root_window):
            self.controller.reset_grid_to_defaults()

    def _save_markov_settings(self):
        """Save Markov settings through controller"""
        self.controller.save_markov_settings(self.markov_enabled_var.get(), self.markov_confidence_var.get())

    def _reset_markov_to_defaults(self):
        """Reset Markov settings to defaults through controller"""
        if messagebox.askyesno("Are you sure you want to reset Markov Chain settings to defaults?", parent=self.root_window):
            self.controller.reset_markov_to_defaults()

    def _save_sound_settings(self):
        """Save Sound Recognizer settings through controller"""
        self.controller.save_sound_settings(self.sound_confidence_var.get(), self.sound_vote_var.get())

    def _reset_sound_to_defaults(self):
        """Reset Sound Recognizer settings to defaults through controller"""
        if messagebox.askyesno(
            "Are you sure you want to reset Sound Recognizer settings to defaults?",
            parent=self.root_window,
        ):
            self.controller.reset_sound_to_defaults()

    def _save_voice_settings(self):
        """Save Voice settings through controller"""
        self.controller.save_voice_settings(self.dictation_silent_chunks_var.get(), self.command_silent_chunks_var.get())

    def _reset_voice_to_defaults(self):
        """Reset Voice settings to defaults through controller"""
        if messagebox.askyesno(
            "Are you sure you want to reset Voice settings to defaults?",
            parent=self.root_window,
        ):
            self.controller.reset_voice_to_defaults()

    def refresh_settings(self):
        """Refresh settings display"""
        if not self._is_alive:
            return
        self._load_current_settings()
        self._force_entry_updates()

    def destroy(self):
        """Clean up resources when view is destroyed"""
        self._is_alive = False
        self.controller.set_view_callback(None)
        super().destroy()
