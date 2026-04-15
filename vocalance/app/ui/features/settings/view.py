import asyncio
import logging
import threading
from functools import partial
from typing import Any, Dict, Optional, Tuple

from PySide6.QtCore import Signal, Slot
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QMessageBox, QVBoxLayout, QWidget

from vocalance.app.config.app_config import DEFAULT_LLM_MODEL_ID, get_whitelisted_llm_model, local_llm_allowlist
from vocalance.app.ui.components.buttons import DangerButton, PrimaryButton
from vocalance.app.ui.components.checkboxes import Checkbox
from vocalance.app.ui.components.inputs import TextInput
from vocalance.app.ui.components.labels import BoxTitleLabel, SectionTitle, SmallLabel
from vocalance.app.ui.components.layouts import Box, FormField, ScrollableContainer
from vocalance.app.ui.features.settings.llm_download_dialog import LlmDownloadProgressDialog
from vocalance.app.ui.qt_theme import theme


class QtSettingsView(QWidget):
    """Qt-based settings view.

    Features:
    - Display all settings in sections
    - Save settings per section
    - Reset to defaults per section
    - Real-time updates from controller
    """

    _llm_persist_finished = Signal(bool, str)

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize settings view."""
        super().__init__(parent)

        self.logger = logging.getLogger(self.__class__.__name__)
        self._llm_persist_finished.connect(self._on_llm_persist_finished_main)
        self.controller = None
        self.settings: Dict[str, Any] = {}
        self.setting_widgets = {}
        self.section_widgets = {}  # Track widgets per section
        self._suppress_llm_combo_events = False
        self._llm_download_busy = False

        self._setup_ui()
        self.logger.debug("QtSettingsView initialized")

    def set_controller(self, controller) -> None:
        """Set the controller and connect signals."""
        self.controller = controller
        self._pending_save_section = None  # Track which section is being saved

        # Connect controller signals
        self.controller.settings_loaded.connect(self._on_settings_loaded)
        self.controller.setting_changed.connect(self._on_setting_changed)
        self.controller.all_settings_changed.connect(self._on_all_settings_changed)
        self.controller.settings_reset.connect(self._on_settings_reset)
        self.controller.operation_error.connect(self._on_error)

        # Load initial settings
        self.logger.info("Loading settings from controller")
        self.controller.load_settings()

    def _setup_ui(self) -> None:
        """Build single-box layout with per-section controls."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(
            theme.config.spacing.large, theme.config.spacing.large, theme.config.spacing.large, theme.config.spacing.large
        )
        main_layout.setSpacing(theme.config.spacing.large)

        # Single box for settings
        self.settings_box = Box(layout="vertical")

        # Title
        self.settings_box.add(BoxTitleLabel(text="Settings"))

        # Scrollable settings area
        self.scroll_container = ScrollableContainer()
        self.settings_box.add(self.scroll_container, stretch=1)

        main_layout.addWidget(self.settings_box)

    def _on_settings_loaded(self, settings: Dict[str, Any]) -> None:
        """Handle settings loaded from controller."""
        try:
            self.settings = settings
            self._refresh_settings_display()
            self.logger.info(f"Settings loaded: {len(settings)} items")
        except Exception as e:
            self.logger.error(f"Error loading settings: {e}", exc_info=True)
            self._show_error(f"Error loading settings: {e}")

    def _on_setting_changed(self, key: str, value: Any) -> None:
        """Handle individual setting changed event."""
        self.settings[key] = value
        self.logger.debug(f"Setting changed: {key} = {value}")

    def _on_all_settings_changed(self, settings: Dict[str, Any]) -> None:
        """Handle all settings changed event."""
        self.settings = settings
        self._refresh_settings_display()
        self.logger.info("All settings updated")

    def _on_settings_reset(self) -> None:
        """Handle settings reset event from controller."""
        if self.controller:
            self.controller.load_settings()
        self.logger.info("Settings reset event received")
        # Note: specific section messages are shown by the reset button handlers

    def _on_error(self, error_msg: str) -> None:
        """Handle error from controller."""
        self._pending_save_section = None  # Clear pending save on error
        self._show_error(error_msg)

    def _refresh_settings_display(self) -> None:
        """Refresh the settings display with organized sections and per-section controls."""
        self.scroll_container.clear_content()

        self.setting_widgets.clear()
        self.section_widgets.clear()

        self._build_llm_model_section()

        # Define which settings to display
        visible_settings = {
            "Grid Settings": [
                ("grid", "default_rect_count", "Default Cell Count"),
            ],
            "Sound Recognizer Settings": [
                ("sound_recognizer", "confidence_threshold", "Confidence Threshold"),
                ("sound_recognizer", "vote_threshold", "Vote Threshold"),
            ],
            "Voice Settings": [
                ("vad", "command_silent_chunks_for_end", "Max Silent Command Chunks"),
            ],
        }

        # Display each section with filtered settings
        for section_name, field_specs in visible_settings.items():
            section_widgets_dict = {}

            # Section title
            self.scroll_container.add(SectionTitle(section_name))

            # Section items
            for category, key, label_text in field_specs:
                # Get value from settings
                value = None
                if category in self.settings and isinstance(self.settings[category], dict):
                    value = self.settings[category].get(key)

                if value is None:
                    continue

                setting_key = f"{category}.{key}"

                # Create widgets based on type
                # CRITICAL: Check bool BEFORE int/float because isinstance(True, int) == True in Python!
                if isinstance(value, bool):
                    checkbox = Checkbox(
                        text=label_text,
                        checked=value,
                        command=lambda state, k=setting_key: None,  # Don't save on change
                    )
                    self.scroll_container.add(checkbox)
                    self.setting_widgets[setting_key] = checkbox
                    section_widgets_dict[setting_key] = checkbox

                elif isinstance(value, (int, float, str)):
                    inp = TextInput()
                    inp.setText(str(value))

                    group = FormField(label_text, inp)
                    self.scroll_container.add(group)
                    self.setting_widgets[setting_key] = inp
                    section_widgets_dict[setting_key] = inp

            # Store section widgets mapping
            self.section_widgets[section_name] = {"fields": field_specs, "widgets": section_widgets_dict}

            # Add button row for this section
            button_layout = QHBoxLayout()
            button_layout.setSpacing(theme.config.spacing.medium)

            # Save button - use partial to avoid lambda closure issues
            save_btn = PrimaryButton(text="Save", command=partial(self._on_save_section_clicked, section_name))
            button_layout.addWidget(save_btn)

            # Reset to defaults button - use partial to avoid lambda closure issues
            reset_btn = DangerButton(text="Reset to Defaults", command=partial(self._on_reset_section_clicked, section_name))
            button_layout.addWidget(reset_btn)

            button_layout.addStretch()

            # Add button row to scroll container
            self.scroll_container.content_layout.addLayout(button_layout)

            # Add spacing between sections
            self.scroll_container.content_layout.addSpacing(theme.config.spacing.large)

        # Add stretch at end
        self.scroll_container.add_stretch()

    def _build_llm_model_section(self) -> None:
        """Whitelisted Qwen GGUF selection, download, and numeric LLM parameters."""
        section_name = "LLM model"
        self.scroll_container.add(SectionTitle(section_name))

        llm_widgets: dict = {}
        field_specs = [
            ("llm", "selected_model_id", "Model"),
            ("llm", "context_length", "Max Context Tokens"),
            ("llm", "max_tokens", "Max Output Tokens"),
        ]

        combo = QComboBox(self)
        self._apply_llm_model_combo_style(combo)
        for m in local_llm_allowlist().artifacts:
            combo.addItem(m.label, m.id)
        current_id = DEFAULT_LLM_MODEL_ID
        if "llm" in self.settings and isinstance(self.settings["llm"], dict):
            current_id = self.settings["llm"].get("selected_model_id") or current_id
        combo.currentIndexChanged.connect(self._on_llm_model_combo_changed)
        self._suppress_llm_combo_events = True
        try:
            idx = combo.findData(current_id)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        finally:
            self._suppress_llm_combo_events = False

        group = FormField("Model", combo)
        self.scroll_container.add(group)
        llm_widgets["llm.selected_model_id"] = combo
        self.setting_widgets["llm.selected_model_id"] = combo

        self._llm_availability_label = SmallLabel("", color=theme.config.text.medium)
        self._llm_availability_label.setWordWrap(True)
        self.scroll_container.add(self._llm_availability_label)

        download_row = QHBoxLayout()
        download_row.setSpacing(theme.config.spacing.medium)
        self._llm_download_btn = PrimaryButton(text="Download", command=self._on_llm_download_clicked)
        download_row.addWidget(self._llm_download_btn)
        download_row.addStretch()
        self.scroll_container.content_layout.addLayout(download_row)

        for category, key, label_text in field_specs[1:]:
            value = None
            if category in self.settings and isinstance(self.settings[category], dict):
                value = self.settings[category].get(key)
            if value is None:
                continue
            setting_key = f"{category}.{key}"
            inp = TextInput()
            inp.setText(str(value))
            group = FormField(label_text, inp)
            self.scroll_container.add(group)
            llm_widgets[setting_key] = inp
            self.setting_widgets[setting_key] = inp

        button_layout = QHBoxLayout()
        button_layout.setSpacing(theme.config.spacing.medium)
        save_btn = PrimaryButton(text="Save", command=partial(self._on_save_section_clicked, section_name))
        button_layout.addWidget(save_btn)
        reset_btn = DangerButton(text="Reset to Defaults", command=partial(self._on_reset_section_clicked, section_name))
        button_layout.addWidget(reset_btn)
        button_layout.addStretch()
        self.scroll_container.content_layout.addLayout(button_layout)
        self.scroll_container.content_layout.addSpacing(theme.config.spacing.large)

        self.section_widgets[section_name] = {"fields": field_specs, "widgets": llm_widgets}
        self._sync_llm_model_ui_state()

    def _apply_llm_model_combo_style(self, combo: QComboBox) -> None:
        combo.setFont(theme.get_font(size="medium"))

    def _set_llm_download_busy(self, busy: bool) -> None:
        self._llm_download_busy = busy
        btn = getattr(self, "_llm_download_btn", None)
        if btn:
            btn.setEnabled(not busy)

    def _set_llm_status_label_muted(self, lbl: SmallLabel) -> None:
        pal = lbl.palette()
        pal.setColor(QPalette.ColorRole.WindowText, QColor(theme.config.text.medium))
        lbl.setPalette(pal)

    def _sync_llm_model_ui_state(self) -> None:
        if not self.controller:
            return
        combo = self.setting_widgets.get("llm.selected_model_id")
        lbl = getattr(self, "_llm_availability_label", None)
        btn = getattr(self, "_llm_download_btn", None)
        if not isinstance(combo, QComboBox) or lbl is None or btn is None:
            return
        raw = combo.currentData()
        combo_mid = str(raw) if raw else DEFAULT_LLM_MODEL_ID
        llm_settings = self.settings.get("llm") if isinstance(self.settings.get("llm"), dict) else {}
        eff = str(llm_settings.get("selected_model_id") or DEFAULT_LLM_MODEL_ID)
        eff_ok = self.controller.llm_bundle_on_disk(eff)
        combo_ok = self.controller.llm_bundle_on_disk(combo_mid)

        if self._llm_download_busy:
            self._set_llm_status_label_muted(lbl)
            lbl.setText("Download in progress…")
            lbl.setVisible(True)
            btn.setVisible(True)
            return

        if eff_ok and combo_ok:
            lbl.clear()
            lbl.setVisible(False)
            btn.setVisible(False)
            return

        self._set_llm_status_label_muted(lbl)
        lbl.setVisible(True)
        btn.setVisible(True)
        if not eff_ok:
            lbl.setText("The saved model is not fully on this device. Download it, or choose another installed model.")
        else:
            lbl.setText("This model is not on this device. Download to use it; until then the saved model stays active.")

    def _on_llm_model_combo_changed(self, _index: int) -> None:
        if getattr(self, "_suppress_llm_combo_events", False) or not self.controller:
            return
        combo = self.setting_widgets.get("llm.selected_model_id")
        if not isinstance(combo, QComboBox):
            return
        model_id = combo.currentData()
        if not model_id:
            return
        mid = str(model_id)
        if self.controller.llm_bundle_on_disk(mid):
            self.controller.update_setting("llm.selected_model_id", mid)
        self._sync_llm_model_ui_state()

    def _on_llm_download_clicked(self) -> None:
        if not self.controller or self._llm_download_busy:
            return
        combo = self.setting_widgets.get("llm.selected_model_id")
        if not isinstance(combo, QComboBox):
            return
        model_id = combo.currentData()
        if not model_id:
            return
        mid = str(model_id)
        if self.controller.llm_bundle_on_disk(mid):
            self._sync_llm_model_ui_state()
            return

        spec = get_whitelisted_llm_model(mid)
        dlg = LlmDownloadProgressDialog(self, model_label=spec.label if spec else mid)
        cancel_ev = threading.Event()
        dlg.cancel_clicked.connect(cancel_ev.set)

        self._llm_active_download = {"dlg": dlg, "mid": mid, "download_msg": ""}
        self._set_llm_download_busy(True)
        self._sync_llm_model_ui_state()

        self.controller.llm_download_progress.connect(dlg.set_status)
        self.controller.llm_cancellable_download_finished.connect(self._on_llm_cancellable_download_finished)
        try:
            self.controller.schedule_llm_cancellable_download(mid, cancel_ev)
            dlg.exec()
        finally:
            try:
                self.controller.llm_cancellable_download_finished.disconnect(self._on_llm_cancellable_download_finished)
            except TypeError:
                pass

    def _on_llm_cancellable_download_finished(self, ok: bool, msg: str) -> None:
        ctx = getattr(self, "_llm_active_download", None)
        if not ctx or not self.controller:
            self.logger.warning("LLM download finished without active context (ok=%s)", ok)
            return
        dlg = ctx["dlg"]
        mid = ctx["mid"]
        try:
            self.controller.llm_download_progress.disconnect(dlg.set_status)
        except TypeError:
            pass

        self.logger.info("LLM download UI handling finished ok=%s mid=%s", ok, mid)

        if ok:
            ctx["download_msg"] = msg

            async def _persist() -> Tuple[bool, str]:
                return await self.controller.update_setting_async("llm.selected_model_id", mid)

            fut = asyncio.create_task(_persist())
            fut.add_done_callback(self._on_llm_persist_future_done)
        else:
            dlg.apply_outcome(False, msg)
            self._llm_active_download = None
            self._set_llm_download_busy(False)
            self._sync_llm_model_ui_state()
            if msg and "cancel" not in msg.lower():
                QMessageBox.warning(self, "Download failed", msg)

    def _on_llm_persist_future_done(self, fut) -> None:
        try:
            s_ok, s_msg = fut.result()
        except Exception as e:
            s_ok, s_msg = False, str(e)
        self._llm_persist_finished.emit(s_ok, s_msg)

    @Slot(bool, str)
    def _on_llm_persist_finished_main(self, s_ok: bool, s_msg: str) -> None:
        ctx = getattr(self, "_llm_active_download", None)
        if not ctx or not self.controller:
            self._set_llm_download_busy(False)
            self._sync_llm_model_ui_state()
            return
        dlg = ctx["dlg"]
        download_msg = ctx.get("download_msg", "")
        self._llm_active_download = None

        dlg.apply_outcome(s_ok, s_msg if not s_ok else download_msg)
        self._set_llm_download_busy(False)
        self._sync_llm_model_ui_state()
        if s_ok:
            self.controller.load_settings()
        else:
            QMessageBox.warning(self, "Could not activate model", s_msg)
        self.logger.info("LLM download persisted and UI cleared (s_ok=%s)", s_ok)

    def _on_save_section_clicked(self, section_name: str) -> None:
        """Handle save button clicked for a specific section."""
        try:
            if not self.controller:
                return

            # Get widgets for this section
            section_info = self.section_widgets.get(section_name)
            if not section_info:
                self.logger.warning(f"Section not found: {section_name}")
                return

            # Define expected types for each setting
            setting_types = self._get_setting_types()

            # Collect settings to save from this section
            settings_to_save = {}
            for setting_key, widget in section_info["widgets"].items():
                if setting_key == "llm.selected_model_id":
                    continue
                # Get current value from widget
                if isinstance(widget, Checkbox):
                    value = widget.isChecked()
                elif isinstance(widget, QComboBox):
                    value = widget.currentData()
                    if value is None:
                        self._show_error(f"Invalid value for {setting_key}")
                        return
                    value = str(value)
                elif isinstance(widget, TextInput):
                    text_value = widget.text().strip()

                    # Get the expected type for this setting
                    expected_type = setting_types.get(setting_key)
                    if expected_type is None:
                        self._show_error(f"Unknown setting type for {setting_key}")
                        return

                    # Convert text input to the expected type
                    try:
                        if expected_type == int:
                            value = int(text_value)
                        elif expected_type == float:
                            value = float(text_value)
                        elif expected_type == bool:
                            # Handle bool strings
                            if text_value.lower() in ("true", "1", "yes", "on"):
                                value = True
                            elif text_value.lower() in ("false", "0", "no", "off"):
                                value = False
                            else:
                                self._show_error(f"Invalid boolean value for {setting_key}. Use: true/false, yes/no, on/off, 1/0")
                                return
                        else:  # str
                            value = text_value
                    except ValueError:
                        self._show_error(f"Invalid value for {setting_key}. Expected {expected_type.__name__}, got: {text_value}")
                        return
                else:
                    continue

                settings_to_save[setting_key] = value

            # Save all settings in this section
            if settings_to_save:
                # Store section name for success callback
                self._pending_save_section = section_name
                self.controller.update_settings(settings_to_save)
                self.logger.info(f"Saved {len(settings_to_save)} settings from section: {section_name}")

        except Exception as e:
            self.logger.error(f"Error saving section {section_name}: {e}", exc_info=True)
            self._show_error(f"Error saving settings: {e}")

    def _on_reset_section_clicked(self, section_name: str) -> None:
        """Handle reset button clicked for a specific section."""
        reply = QMessageBox.question(
            self,
            "Reset Section",
            f"Reset {section_name} to defaults?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                if not self.controller:
                    return

                # Get settings keys for this section
                section_info = self.section_widgets.get(section_name)
                if not section_info:
                    return

                setting_keys = list(section_info["widgets"].keys())

                # Reset settings for this section
                self.controller.reset_section_settings(setting_keys)
                self.logger.info(f"Reset section to defaults: {section_name}")

                # Show success message
                QMessageBox.information(self, "Success", f"{section_name} reset to defaults!")

            except Exception as e:
                self.logger.error(f"Error resetting section: {e}", exc_info=True)
                self._show_error(f"Error resetting section: {e}")

    def _get_setting_types(self) -> Dict[str, type]:
        """Define the expected type for each setting."""
        return {
            # LLM model
            "llm.selected_model_id": str,
            "llm.context_length": int,
            "llm.max_tokens": int,
            # Grid Settings
            "grid.default_rect_count": int,
            # Sound Recognizer Settings
            "sound_recognizer.confidence_threshold": float,
            "sound_recognizer.vote_threshold": float,
            # Voice Settings
            "vad.command_silent_chunks_for_end": int,
        }

    def _show_error(self, message: str) -> None:
        """Show error message dialog."""
        QMessageBox.critical(self, "Error", message)
