import tkinter as tk
from typing import List

import customtkinter as ctk

from iris.app.ui.controls.sound_control import SoundController
from iris.app.ui.utils.ui_icon_utils import set_window_icon_robust
from iris.app.ui.views.components.base_view import ViewHelper
from iris.app.ui.views.components.form_builder import FormBuilder
from iris.app.ui.views.components.list_builder import ButtonType, ListBuilder, ListItemColumn
from iris.app.ui.views.components.themed_components import PrimaryButton, ThemedLabel, TransparentFrame, TwoColumnTabLayout
from iris.app.ui.views.components.view_config import view_config


class SoundView(ViewHelper):
    """Simplified sound view using base components and form builder"""

    def __init__(self, parent, controller: SoundController, root_window):
        super().__init__(parent, controller, root_window)
        self.current_training_sample = 0
        self.total_training_samples = 0
        self._setup_ui()
        self.controller.refresh_sound_list()

    def _setup_ui(self) -> None:
        """Setup the main UI layout"""
        self.setup_main_layout()

        self.layout = TwoColumnTabLayout(self, "Add Sound", "Manage Sounds")
        self.layout.grid(row=0, column=0, sticky="nsew")

        self._setup_training_form()
        self._setup_sounds_list_panel()

    def _setup_training_form(self) -> None:
        """Setup training form using form builder"""
        container = self.layout.left_content
        form_builder = FormBuilder()
        form_builder.setup_form_grid(container)

        # Create form fields
        self.name_label, self.sound_name_entry = form_builder.create_labeled_entry(container, "Sound name:", "Choose a name...")

        self.samples_label, self.samples_entry = form_builder.create_labeled_entry(
            container,
            "Number of samples:",
            view_config.form_defaults.placeholder_samples,
            default_value=view_config.form_defaults.placeholder_samples,
        )

        # Add button - store reference to the button frame
        self.record_button_frame = ctk.CTkFrame(container, fg_color="transparent")
        self.record_button_frame.grid(
            row=4,
            column=0,
            sticky="ew",
            pady=view_config.theme.spacing.small,
            padx=view_config.theme.two_box_layout.inner_content_padx,
        )
        self.record_button_frame.grid_columnconfigure(0, weight=1)

        self.record_button = PrimaryButton(
            self.record_button_frame, text=view_config.theme.button_text.record, command=self._start_training
        )
        self.record_button.grid(row=0, column=0, sticky="ew")

        # Status frame (initially hidden)
        self.training_status_frame = TransparentFrame(container)
        self.training_status_frame.grid_columnconfigure(0, weight=1)

        # Single status label for "Recording sample X of Y"
        self.training_status_label = ThemedLabel(self.training_status_frame, text="")
        self.training_status_label.grid(row=0, column=0, pady=view_config.theme.spacing.small, sticky="ew")

        # Progress bar
        self.training_progress_bar = ctk.CTkProgressBar(
            self.training_status_frame,
            fg_color=view_config.theme.shape_colors.lightest,
            progress_color=view_config.theme.text_colors.medium,
        )
        self.training_progress_bar.set(0)
        self.training_progress_bar.grid(row=1, column=0, pady=view_config.theme.spacing.small, sticky="ew")

        container.grid_rowconfigure(5, weight=1)

    def _setup_sounds_list_panel(self) -> None:
        """Setup sounds list panel"""
        container = self.layout.right_content
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.sounds_scroll_frame = ListBuilder.create_scrollable_list_container(container, row=0, column=0)

        form_builder = FormBuilder()
        form_builder.create_button_row(
            container,
            [{"text": view_config.theme.button_text.delete_all_sounds, "command": self._delete_all_sounds, "type": "danger"}],
            extra_pady=(0, view_config.theme.two_box_layout.last_element_bottom_padding),
            extra_padx=view_config.theme.two_box_layout.box_content_padding,
            row=1,
        )

    def _start_training(self) -> None:
        """Start training with validation"""
        sound_name = self.sound_name_entry.get().strip()
        if not sound_name:
            self.show_error("Error", "Please enter a sound name")
            return

        try:
            num_samples = int(self.samples_entry.get())
            if num_samples <= 0:
                self.show_error("Error", "Number of samples must be positive")
                return
        except (ValueError, TypeError):
            self.show_error("Error", "Please enter a valid number of samples")
            return

        self._toggle_training_interface(training_active=True)
        self.controller.train_sound(sound_name, num_samples)

    def _toggle_training_interface(self, training_active: bool) -> None:
        """Toggle between training and input interface"""
        input_widgets = [self.name_label, self.sound_name_entry, self.samples_label, self.samples_entry, self.record_button_frame]

        for widget in input_widgets:
            if training_active:
                widget.grid_remove()
            else:
                widget.grid()

        if training_active:
            self.training_status_frame.grid(
                row=4,
                column=0,
                sticky="nsew",
                padx=view_config.theme.two_box_layout.inner_content_padx,
                pady=view_config.theme.spacing.medium,
            )
        else:
            self.training_status_frame.grid_remove()

    def reset_training_interface(self) -> None:
        """Reset training interface to initial state"""
        self.current_training_sample = 0
        self.total_training_samples = 0
        # Show progress bar again for next training
        self.safe_widget_operation(lambda: self.training_progress_bar.grid())
        self.safe_widget_operation(lambda: self.training_progress_bar.set(0))
        self.safe_widget_operation(lambda: self.training_status_label.configure(text=""))
        self._toggle_training_interface(training_active=False)
        self.clear_form_fields(self.sound_name_entry, self.samples_entry)
        self.samples_entry.insert(0, view_config.form_defaults.placeholder_samples)

    def display_sounds(self, sounds: List[str]) -> None:
        """Display sounds in the list"""
        if not self.safe_widget_operation(lambda: self.sounds_scroll_frame.winfo_exists()):
            self.schedule_delayed_action(lambda: self.display_sounds(sounds))
            return

        ListBuilder.display_items(
            container=self.sounds_scroll_frame,
            items=sounds,
            create_item_callback=self._create_sound_item,
            empty_message="No available sounds.\nUse the left panel to record a sound.",
        )

    def _create_sound_item(self, sound_name: str, row_index: int) -> None:
        """Create a sound item in the list"""
        try:
            mapped_command = self.controller.get_sound_command_mapping(sound_name)
            mapping_status = mapped_command if mapped_command else "Unmapped"
        except Exception as e:
            self.logger.debug(f"Could not get command mapping for {sound_name}: {e}")
            mapping_status = "Unmapped"

        status_color = (
            view_config.theme.text_colors.medium if mapping_status == "Unmapped" else view_config.theme.accent_colors.success_text
        )

        ListBuilder.create_list_item(
            container=self.sounds_scroll_frame,
            row_index=row_index,
            columns=[
                ListItemColumn.label(text=sound_name, weight=1),
                ListItemColumn.label(text=mapping_status, weight=0, anchor="e", color=status_color),
                ListItemColumn.button(
                    text=view_config.theme.button_text.map,
                    command=lambda s=sound_name: self._map_sound_to_command(s),
                    button_type=ButtonType.PRIMARY,
                ),
                ListItemColumn.button(
                    text=view_config.theme.button_text.delete,
                    command=lambda s=sound_name: self._delete_sound(s),
                    button_type=ButtonType.DANGER,
                ),
            ],
        )

    def _map_sound_to_command(self, sound_name: str) -> None:
        """Show dialog to map sound to command with proper dropdown filtering and themed UI"""
        # Get theme
        theme = view_config.theme
        label_font_bold = (theme.font_family.primary, theme.font_sizes.medium, "bold")
        dropdown_font = (theme.font_family.primary, theme.font_sizes.medium)

        dialog = ctk.CTkToplevel(self.root_window)
        dialog.title(f"Map Sound: {sound_name}")
        dialog.transient(self.root_window)
        dialog.grab_set()
        dialog.configure(fg_color=theme.shape_colors.darkest)
        dialog.minsize(theme.dimensions.sound_mapping_dialog_width, theme.dimensions.sound_mapping_dialog_min_height)

        try:
            set_window_icon_robust(dialog)
        except Exception:
            pass

        # Create main frame with themed background
        main_frame = ctk.CTkFrame(dialog, fg_color=theme.shape_colors.dark, border_color=theme.shape_colors.medium, border_width=1)
        main_frame.grid(
            row=0,
            column=0,
            sticky="ew",
            padx=theme.two_box_layout.inner_content_padx,
            pady=theme.two_box_layout.inner_content_padx,
        )

        dialog.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        # Get available command types and initialize variables
        command_types = self.controller.get_mapping_command_types()
        type_var = tk.StringVar(value=command_types[0] if command_types else "Commands")
        value_var = tk.StringVar()

        # Command type dropdown
        type_label = ThemedLabel(main_frame, text="Command Type:", font=label_font_bold)
        type_label.grid(
            row=0,
            column=0,
            sticky="w",
            pady=(theme.spacing.medium, theme.spacing.tiny),
            padx=theme.two_box_layout.inner_content_padx,
        )

        type_dropdown = ctk.CTkOptionMenu(
            main_frame,
            values=command_types,
            variable=type_var,
            command=self._on_command_type_changed,
            fg_color=theme.shape_colors.darkest,
            button_color=theme.shape_colors.darkest,
            button_hover_color=theme.shape_colors.medium,
            text_color=theme.text_colors.light,
            font=dropdown_font,
            dropdown_font=dropdown_font,
            height=32,
            dropdown_fg_color=theme.shape_colors.darkest,
            dropdown_hover_color=theme.shape_colors.medium,
            dropdown_text_color=theme.text_colors.light,
        )
        type_dropdown.grid(
            row=1, column=0, sticky="ew", pady=(0, theme.spacing.small), padx=theme.two_box_layout.inner_content_padx
        )

        # Command value dropdown
        value_label = ThemedLabel(main_frame, text="Command Value:", font=label_font_bold)
        value_label.grid(
            row=2,
            column=0,
            sticky="w",
            pady=(theme.spacing.medium, theme.spacing.tiny),
            padx=theme.two_box_layout.inner_content_padx,
        )

        value_dropdown = ctk.CTkOptionMenu(
            main_frame,
            values=[],
            variable=value_var,
            fg_color=theme.shape_colors.darkest,
            button_color=theme.shape_colors.darkest,
            button_hover_color=theme.shape_colors.medium,
            text_color=theme.text_colors.light,
            font=dropdown_font,
            dropdown_hover_color=theme.shape_colors.medium,
            dropdown_font=dropdown_font,
            height=32,
            dropdown_fg_color=theme.shape_colors.darkest,
            dropdown_text_color=theme.text_colors.light,
        )
        value_dropdown.grid(
            row=3, column=0, sticky="ew", pady=(0, theme.spacing.medium), padx=theme.two_box_layout.inner_content_padx
        )

        # Store references for the callback
        self._temp_dialog_refs = {"type_var": type_var, "value_var": value_var, "value_dropdown": value_dropdown}

        def on_confirm():
            command_type = type_var.get()
            command_value = value_var.get().strip()

            if command_value:
                # Create appropriate command phrase based on type
                if command_type == "Commands":
                    command_phrase = command_value
                elif command_type == "Marks":
                    command_phrase = f"mark:{command_value}"
                elif command_type == "Grid":
                    command_phrase = command_value
                else:
                    command_phrase = command_value

                self.controller.map_sound_to_command(sound_name, command_phrase)
                dialog.destroy()
                self.refresh_sounds_list()

                # Clean up temp references
                if hasattr(self, "_temp_dialog_refs"):
                    delattr(self, "_temp_dialog_refs")

        # Use FormBuilder to create the confirm button row at the bottom (after the dropdowns)
        form_builder = FormBuilder()
        form_builder.create_button_row(
            main_frame,
            [
                {"text": theme.button_text.confirm, "command": on_confirm, "type": "primary"},
            ],
            row=4,  # Place after the dropdowns (rows 0-3)
        )

        # Initialize the value dropdown with the default command type
        self._on_command_type_changed(type_var.get())

    def _on_command_type_changed(self, selected_type: str) -> None:
        """Handle command type dropdown change to update available values"""
        if not hasattr(self, "_temp_dialog_refs"):
            return

        value_dropdown = self._temp_dialog_refs["value_dropdown"]
        value_var = self._temp_dialog_refs["value_var"]

        # Get appropriate values based on selected type
        if selected_type == "Commands":
            values = self.controller.get_available_exact_match_commands()
        elif selected_type == "Marks":
            values = self.controller.get_available_mark_names()
        elif selected_type == "Grid":
            values = self.controller.get_grid_trigger_words()
        else:
            values = []

        # Update dropdown values
        if values:
            value_dropdown.configure(values=values)
            value_var.set(values[0])
        else:
            value_dropdown.configure(values=["No options available"])
            value_var.set("No options available")

    def _delete_sound(self, sound_name: str) -> None:
        """Delete a sound"""
        self.controller.delete_individual_sound(sound_name)

    def _delete_all_sounds(self) -> None:
        """Delete all sounds"""
        if self.show_delete_all_confirmation("sounds"):
            self.controller.delete_all_sounds()

    def refresh_sounds_list(self) -> None:
        """Refresh the sounds list"""
        self.controller.refresh_sound_list()

    # Training status update methods
    def update_training_status(self, status: str, progress: float = 0.0) -> None:
        """Update training status display with simple message and progress"""
        self.safe_widget_operation(lambda: self.training_status_label.configure(text=status))
        self.safe_widget_operation(lambda: self.training_progress_bar.set(progress))

    def update_training_progress(self, sound_name: str, status: str, current_sample: int, total_samples: int) -> None:
        """Update training progress"""
        self.current_training_sample = current_sample
        self.total_training_samples = total_samples

        progress_value = current_sample / total_samples if total_samples > 0 else 0
        status_text = f"Recording sample {current_sample} of {total_samples}"
        self.update_training_status(status_text, progress_value)

    # Controller callback methods
    def on_sounds_updated(self, sounds: List[str]) -> None:
        """Handle sounds list updated"""
        self.display_sounds(sounds)

    def on_training_initiated(self, sound_name: str, total_samples: int) -> None:
        """Handle training initiated"""
        self.total_training_samples = total_samples
        self.current_training_sample = 0
        self.update_training_status(f"Recording sample 1 of {total_samples}", 0.0)

    def on_sample_recorded(self, current_sample: int, total_samples: int, is_last: bool) -> None:
        """Handle sample recorded"""
        self.current_training_sample = current_sample
        progress_value = current_sample / total_samples

        if is_last:
            # Hide progress bar and show 'Training...' message
            self.safe_widget_operation(lambda: self.training_progress_bar.grid_remove())
            self.update_training_status("Training...", 0.0)
        else:
            next_sample = current_sample + 1
            self.safe_widget_operation(lambda: self.training_progress_bar.grid())
            self.update_training_status(f"Recording sample {next_sample} of {total_samples}", progress_value)

    def on_training_progress(self, label: str, current_sample: int, total_samples: int) -> None:
        """Handle training progress updates"""
        progress_value = current_sample / total_samples if total_samples > 0 else 0.0
        self.safe_widget_operation(lambda: self.training_progress_bar.grid())
        self.update_training_status(f"Training '{label}': {current_sample}/{total_samples} samples", progress_value)

    def on_training_complete(self, sound_name: str) -> None:
        """Handle training complete"""
        self.refresh_sounds_list()
        # Reset interface immediately after a short delay
        self.schedule_delayed_action(self.reset_training_interface, 1000)

    def on_training_failed(self, sound_name: str, reason: str) -> None:
        """Handle training failed"""
        self.update_training_status("Training failed", 0.0)
        self.schedule_delayed_action(self.reset_training_interface, 2000)

    def on_training_status(self, message: str, status_type: str) -> None:
        """Handle training status updates"""
        # Simplified - only update status label with the message
        self.safe_widget_operation(lambda: self.training_status_label.configure(text=message))
