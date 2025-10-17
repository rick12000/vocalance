import customtkinter as ctk

from iris.app.ui import ui_theme
from iris.app.ui.controls.settings_control import SettingsController
from iris.app.ui.views.components import themed_dialogs as messagebox
from iris.app.ui.views.components.themed_components import (
    BorderlessFrame,
    BoxTitle,
    DangerButton,
    PrimaryButton,
    ThemedEntry,
    ThemedLabel,
    ThemedScrollableFrame,
    TransparentFrame,
)


class SettingsView(ctk.CTkFrame):
    """UI view for settings tab - handles LLM settings configuration"""

    def __init__(self, parent_frame, controller: SettingsController, root_window):
        super().__init__(parent_frame, fg_color=ui_theme.theme.shape_colors.darkest)

        self.controller = controller
        self.parent_frame = parent_frame
        self.root_window = root_window
        self._is_alive = True

        self.llm_context_length_var = ctk.StringVar()
        self.llm_max_tokens_var = ctk.StringVar()
        self.grid_default_cells_var = ctk.StringVar()
        self.markov_confidence_var = ctk.StringVar()
        self.sound_confidence_var = ctk.StringVar()
        self.sound_vote_var = ctk.StringVar()

        self._build_tab_ui()
        self._load_current_settings()

        self.controller.set_view_callback(self)

    def _create_settings_section(self, parent, title, row, fields, save_command, reset_command):
        """
        Create a reusable settings section with fields and buttons.

        Args:
            parent: Parent frame
            title: Section title
            row: Grid row position
            fields: List of (label, variable, description) tuples
            save_command: Save button command
            reset_command: Reset button command
        """
        frame = BorderlessFrame(
            parent,
            fg_color=ui_theme.theme.shape_colors.dark,
            corner_radius=ui_theme.theme.two_box_layout.box_corner_radius,
        )
        frame.grid(
            row=row,
            column=0,
            sticky="ew",
            padx=(ui_theme.theme.two_box_layout.outer_padding_left, ui_theme.theme.two_box_layout.outer_padding_right),
            pady=(
                ui_theme.theme.spacing.small if row > 0 else ui_theme.theme.spacing.medium,
                ui_theme.theme.spacing.small if row < 3 else ui_theme.theme.spacing.medium,
            ),
        )

        header = BoxTitle(frame, text=title)
        header.grid(
            row=0,
            column=0,
            columnspan=3,
            padx=ui_theme.theme.spacing.medium,
            pady=(ui_theme.theme.spacing.medium, ui_theme.theme.spacing.small),
            sticky="w",
        )

        entries = []
        for idx, (label_text, variable, description) in enumerate(fields, start=1):
            ThemedLabel(frame, text=label_text, bold=True).grid(
                row=idx, column=0, padx=ui_theme.theme.spacing.medium, pady=ui_theme.theme.spacing.small, sticky="w"
            )

            entry = ThemedEntry(frame, textvariable=variable, width=ui_theme.theme.dimensions.entry_width_small)
            entry.grid(row=idx, column=1, padx=ui_theme.theme.spacing.medium, pady=ui_theme.theme.spacing.small, sticky="w")
            entries.append(entry)

            ThemedLabel(frame, text=description, color=ui_theme.theme.text_colors.medium).grid(
                row=idx, column=2, padx=ui_theme.theme.spacing.medium, pady=ui_theme.theme.spacing.small, sticky="w"
            )

        frame.grid_columnconfigure(1, weight=1)
        frame.grid_columnconfigure(2, weight=2)

        buttons_frame = TransparentFrame(frame)
        buttons_frame.grid(row=len(fields) + 1, column=0, columnspan=3, pady=20, padx=20, sticky="ew")
        buttons_frame.grid_columnconfigure(0, weight=0)
        buttons_frame.grid_columnconfigure(1, weight=0)

        PrimaryButton(buttons_frame, text=f"Save {title}", command=save_command).grid(
            row=0, column=0, padx=(0, ui_theme.theme.spacing.small), sticky="w"
        )
        DangerButton(buttons_frame, text="Reset to Defaults", command=reset_command).grid(row=0, column=1, sticky="w")

        return entries

    def _build_tab_ui(self):
        """Build the settings tab UI with LLM settings only"""
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        scrollable_frame = ThemedScrollableFrame(self)
        scrollable_frame.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(ui_theme.theme.header_layout.frame_padding_left, ui_theme.theme.header_layout.frame_padding_right),
            pady=ui_theme.theme.spacing.small,
        )

        for i in range(4):
            scrollable_frame.grid_rowconfigure(i, weight=0)
        scrollable_frame.grid_columnconfigure(0, weight=1)

        # LLM Settings Section
        self._create_settings_section(
            scrollable_frame,
            "LLM Model Settings",
            0,
            [
                ("Context Length:", self.llm_context_length_var, "(128-32768, higher = more context)"),
                ("Max Tokens:", self.llm_max_tokens_var, "(1-1024, higher = longer responses)"),
            ],
            self._save_llm_settings,
            self._reset_llm_to_defaults,
        )

        # Grid Settings Section
        self._create_settings_section(
            scrollable_frame,
            "Grid Settings",
            1,
            [("Default Cell Count:", self.grid_default_cells_var, "(100-10000, cells shown when saying 'golf')")],
            self._save_grid_settings,
            self._reset_grid_to_defaults,
        )

        # Markov Chain Settings Section
        self._create_settings_section(
            scrollable_frame,
            "Markov Chain Settings",
            2,
            [("Prediction Confidence:", self.markov_confidence_var, "(0.0-1.0, threshold for using Markov vs speech model)")],
            self._save_markov_settings,
            self._reset_markov_to_defaults,
        )

        # Sound Recognizer Settings Section
        self._create_settings_section(
            scrollable_frame,
            "Sound Recognizer Settings",
            3,
            [
                ("Confidence Threshold:", self.sound_confidence_var, "(0.0-1.0, minimum similarity for recognition)"),
                ("Vote Threshold:", self.sound_vote_var, "(0.0-1.0, minimum vote alignment percentage)"),
            ],
            self._save_sound_settings,
            self._reset_sound_to_defaults,
        )

    def on_settings_updated(self):
        if not self._is_alive:
            return
        self._load_current_settings()

    def on_validation_error(self, title: str, message: str):
        """Handle validation errors from controller"""
        messagebox.showerror(title, message, parent=self.root_window)

    def on_save_success(self, message: str):
        """Handle successful save from controller"""
        messagebox.showinfo("Success", message, parent=self.root_window)

    def on_save_error(self, message: str):
        """Handle save errors from controller"""
        messagebox.showerror("Error", message, parent=self.root_window)

    def on_reset_complete(self):
        """Handle reset completion from controller"""
        self._load_current_settings()
        messagebox.showinfo("Reset Complete", "Settings have been reset to defaults", parent=self.root_window)

    def _load_current_settings(self):
        """Load current settings from controller"""
        try:
            settings = self.controller.load_current_settings()

            if settings:
                llm_settings = settings.get("llm", {})
                self.llm_context_length_var.set(str(llm_settings.get("context_length", 2048)))
                self.llm_max_tokens_var.set(str(llm_settings.get("max_tokens", 512)))

                grid_settings = settings.get("grid", {})
                self.grid_default_cells_var.set(str(grid_settings.get("default_rect_count", 500)))

                markov_settings = settings.get("markov_predictor", {})
                self.markov_confidence_var.set(str(markov_settings.get("confidence_threshold", 0.95)))

                sound_settings = settings.get("sound_recognizer", {})
                self.sound_confidence_var.set(str(sound_settings.get("confidence_threshold", 0.15)))
                self.sound_vote_var.set(str(sound_settings.get("vote_threshold", 0.35)))
            else:
                for var in [
                    self.llm_context_length_var,
                    self.llm_max_tokens_var,
                    self.grid_default_cells_var,
                    self.markov_confidence_var,
                    self.sound_confidence_var,
                    self.sound_vote_var,
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
        if messagebox.askyesno(
            "Reset LLM Settings", "Are you sure you want to reset LLM settings to defaults?", parent=self.root_window
        ):
            self.controller.reset_llm_to_defaults()

    def _save_grid_settings(self):
        """Save Grid settings through controller"""
        self.controller.save_grid_settings(self.grid_default_cells_var.get())

    def _reset_grid_to_defaults(self):
        """Reset Grid settings to defaults through controller"""
        if messagebox.askyesno(
            "Reset Grid Settings", "Are you sure you want to reset Grid settings to defaults?", parent=self.root_window
        ):
            self.controller.reset_grid_to_defaults()

    def _save_markov_settings(self):
        """Save Markov settings through controller"""
        self.controller.save_markov_settings(self.markov_confidence_var.get())

    def _reset_markov_to_defaults(self):
        """Reset Markov settings to defaults through controller"""
        if messagebox.askyesno(
            "Reset Markov Settings", "Are you sure you want to reset Markov Chain settings to defaults?", parent=self.root_window
        ):
            self.controller.reset_markov_to_defaults()

    def _save_sound_settings(self):
        """Save Sound Recognizer settings through controller"""
        self.controller.save_sound_settings(self.sound_confidence_var.get(), self.sound_vote_var.get())

    def _reset_sound_to_defaults(self):
        """Reset Sound Recognizer settings to defaults through controller"""
        if messagebox.askyesno(
            "Reset Sound Settings",
            "Are you sure you want to reset Sound Recognizer settings to defaults?",
            parent=self.root_window,
        ):
            self.controller.reset_sound_to_defaults()

    def refresh_settings(self):
        if not self._is_alive:
            return
        self._load_current_settings()

    def destroy(self):
        self._is_alive = False
        self.controller.set_view_callback(None)
        super().destroy()
