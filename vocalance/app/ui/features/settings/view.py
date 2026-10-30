import asyncio
import logging
from functools import partial
from typing import Any, Dict, Optional, Tuple

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QMessageBox, QVBoxLayout, QWidget

from vocalance.app.config.app_config import (
    DEFAULT_LLM_MODEL_ID,
    get_whitelisted_llm_model,
    local_llm_allowlist,
)
from vocalance.app.services.storage.user_configurable_settings import (
    FIELD_BY_PATH,
    USER_CONFIGURABLE_FIELDS,
    get_config_field_bounds,
)
from vocalance.app.ui.components.buttons import DangerButton, PrimaryButton
from vocalance.app.ui.components.checkboxes import Checkbox
from vocalance.app.ui.components.inputs import TextInput
from vocalance.app.ui.components.labels import BoxTitleLabel, SectionTitle, SmallLabel
from vocalance.app.ui.components.layouts import Box, FormField, ScrollableContainer
from vocalance.app.ui.features.settings.llm_download_dialog import (
    LlmDownloadProgressDialog,
)
from vocalance.app.ui.qt_theme import theme
from vocalance.app.utils.llm_dep_check import llm_deps_available


class QtSettingsView(QWidget):
    """Qt-based settings view.

    Features:
    - Display all settings in sections
    - Save settings per section
    - Reset to defaults per section
    - Real-time updates from controller
    """

    llm_persist_finished = Signal(bool, str)
    endpoint_test_result = Signal(bool, str)

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize settings view."""
        super().__init__(parent)

        self.logger = logging.getLogger(self.__class__.__name__)
        self._llm_enabled = llm_deps_available()
        self.llm_persist_finished.connect(self.on_llm_persist_finished_main)
        self.endpoint_test_result.connect(self.on_llm_endpoint_test_finished, Qt.ConnectionType.QueuedConnection)
        self.controller = None
        self.settings: Dict[str, Any] = {}
        self.setting_widgets = {}
        self.section_widgets = {}  # Track widgets per section
        self.suppress_llm_combo_events = False
        self.llm_download_busy = False

        self.setup_ui()
        self.logger.debug("QtSettingsView initialized")

    def set_controller(self, controller) -> None:
        """Set the controller and connect signals."""
        self.controller = controller
        self.pending_save_section = None  # Track which section is being saved

        # Connect controller signals
        self.controller.settings_loaded.connect(self.on_settings_loaded)
        self.controller.setting_changed.connect(self.on_setting_changed)
        self.controller.all_settings_changed.connect(self.on_all_settings_changed)
        self.controller.operation_error.connect(self.on_error)
        if self._llm_enabled:
            self.controller.llm_bundle_status_updated.connect(self.sync_llm_model_ui_state)

        # Load initial settings
        self.logger.info("Loading settings from controller")
        self.controller.load_settings()

    def setup_ui(self) -> None:
        """Build single-box layout with per-section controls."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(
            theme.config.spacing.large,
            theme.config.spacing.large,
            theme.config.spacing.large,
            theme.config.spacing.large,
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

    def on_settings_loaded(self, settings: Dict[str, Any]) -> None:
        """Handle settings loaded from controller."""
        self.settings = settings
        self.refresh_settings_display()
        self.logger.info("Settings loaded: %s items", len(settings))

    def on_setting_changed(self, key: str, value: Any) -> None:
        """Handle individual setting changed event."""
        self.settings[key] = value
        self.logger.debug(f"Setting changed: {key} = {value}")

    def on_all_settings_changed(self, settings: Dict[str, Any]) -> None:
        """Handle all settings changed event."""
        self.settings = settings
        self.refresh_settings_display()
        self.logger.info("All settings updated")

    def on_error(self, error_msg: str) -> None:
        """Handle error from controller."""
        self.pending_save_section = None  # Clear pending save on error
        self.show_error(error_msg)

    def refresh_settings_display(self) -> None:
        """Refresh the settings display with organized sections and per-section controls."""
        self.scroll_container.clear_content()

        self.setting_widgets.clear()
        self.section_widgets.clear()

        if self._llm_enabled:
            self.build_llm_model_section()

        sections: Dict[str, Dict] = {}
        for field in USER_CONFIGURABLE_FIELDS:
            if not field.path.startswith("llm."):
                sections.setdefault(field.section, {})[field.path] = field

        for section_name, fields_by_path in sections.items():
            section_widgets_dict = {}

            self.scroll_container.add(SectionTitle(section_name))

            for path, field in fields_by_path.items():
                category, key = path.split(".", 1)
                value = None
                if category in self.settings and isinstance(self.settings[category], dict):
                    value = self.settings[category].get(key)

                if value is None:
                    continue

                if field.choices:
                    widget = QComboBox(self)
                    for stored, label in field.choices:
                        widget.addItem(label, stored)
                    selected = widget.findData(str(value))
                    if selected >= 0:
                        widget.setCurrentIndex(selected)
                else:
                    widget = TextInput()
                    widget.setText(str(value))
                self.scroll_container.add(FormField(field.label, widget))
                self.setting_widgets[path] = widget
                section_widgets_dict[path] = widget

            self.section_widgets[section_name] = {"widgets": section_widgets_dict}

            button_layout = QHBoxLayout()
            button_layout.setSpacing(theme.config.spacing.medium)
            save_btn = PrimaryButton(text="Save", command=partial(self.on_save_section_clicked, section_name))
            button_layout.addWidget(save_btn)
            reset_btn = DangerButton(
                text="Reset to Defaults",
                command=partial(self.on_reset_section_clicked, section_name),
            )
            button_layout.addWidget(reset_btn)
            button_layout.addStretch()
            self.scroll_container.content_layout.addLayout(button_layout)
            self.scroll_container.content_layout.addSpacing(theme.config.spacing.large)

        self.scroll_container.add_stretch()

    def build_llm_model_section(self) -> None:
        section_name = "LLM model"
        self.scroll_container.add(SectionTitle(section_name))

        llm_widgets: dict = {}
        llm_settings = self.settings.get("llm") if isinstance(self.settings.get("llm"), dict) else {}
        current_provider = str(llm_settings.get("provider") or "shipped")

        provider_combo = QComboBox(self)
        provider_field = FIELD_BY_PATH["llm.provider"]
        for stored, label in provider_field.choices:
            provider_combo.addItem(label, stored)
        idx = provider_combo.findData(current_provider)
        if idx >= 0:
            provider_combo.setCurrentIndex(idx)
        provider_combo.currentIndexChanged.connect(self.on_llm_provider_changed)
        self.scroll_container.add(FormField("Inference", provider_combo))
        llm_widgets["llm.provider"] = provider_combo
        self.setting_widgets["llm.provider"] = provider_combo

        self.llm_shipped_panel = QWidget(self)
        shipped_layout = QVBoxLayout(self.llm_shipped_panel)
        shipped_layout.setContentsMargins(0, 0, 0, 0)
        shipped_layout.setSpacing(theme.config.spacing.medium)

        combo = QComboBox(self)
        self.apply_llm_model_combo_style(combo)
        for m in local_llm_allowlist().artifacts:
            combo.addItem(m.label, m.id)
        current_id = DEFAULT_LLM_MODEL_ID
        current_id = llm_settings.get("selected_model_id") or current_id
        combo.currentIndexChanged.connect(self.on_llm_model_combo_changed)
        self.suppress_llm_combo_events = True
        try:
            midx = combo.findData(current_id)
            if midx >= 0:
                combo.setCurrentIndex(midx)
        finally:
            self.suppress_llm_combo_events = False
        shipped_layout.addWidget(FormField("Model", combo))
        llm_widgets["llm.selected_model_id"] = combo
        self.setting_widgets["llm.selected_model_id"] = combo

        self.llm_availability_label = SmallLabel("", color=theme.config.text.medium)
        self.llm_availability_label.setWordWrap(True)
        shipped_layout.addWidget(self.llm_availability_label)

        download_row = QHBoxLayout()
        download_row.setSpacing(theme.config.spacing.medium)
        self.llm_download_btn = PrimaryButton(text="Download", command=self.on_llm_download_clicked)
        download_row.addWidget(self.llm_download_btn)
        download_row.addStretch()
        shipped_layout.addLayout(download_row)

        for setting_key in ("llm.context_length", "llm.max_tokens"):
            field = FIELD_BY_PATH[setting_key]
            category, key = setting_key.split(".", 1)
            value = llm_settings.get(key)
            if value is None:
                continue
            inp = TextInput()
            inp.setText(str(value))
            shipped_layout.addWidget(FormField(field.label, inp))
            llm_widgets[setting_key] = inp
            self.setting_widgets[setting_key] = inp

        self.scroll_container.add(self.llm_shipped_panel)

        self.llm_endpoint_panel = QWidget(self)
        endpoint_layout = QVBoxLayout(self.llm_endpoint_panel)
        endpoint_layout.setContentsMargins(0, 0, 0, 0)
        endpoint_layout.setSpacing(theme.config.spacing.medium)

        for setting_key in ("llm.local_endpoint_base_url", "llm.local_endpoint_model"):
            field = FIELD_BY_PATH[setting_key]
            _, key = setting_key.split(".", 1)
            inp = TextInput()
            inp.setText(str(llm_settings.get(key) or ""))
            endpoint_layout.addWidget(FormField(field.label, inp))
            llm_widgets[setting_key] = inp
            self.setting_widgets[setting_key] = inp

        endpoint_hint = SmallLabel(
            "Loopback only (localhost / 127.0.0.1 / ::1). Compatible with Ollama and llama-server OpenAI APIs.",
            color=theme.config.text.medium,
        )
        endpoint_hint.setWordWrap(True)
        endpoint_layout.addWidget(endpoint_hint)

        test_row = QHBoxLayout()
        self.llm_endpoint_test_btn = PrimaryButton(text="Test connection", command=self.on_llm_endpoint_test_clicked)
        test_row.addWidget(self.llm_endpoint_test_btn)
        test_row.addStretch()
        endpoint_layout.addLayout(test_row)
        self.llm_endpoint_test_status = SmallLabel("", color=theme.config.text.medium)
        self.llm_endpoint_test_status.setWordWrap(True)
        endpoint_layout.addWidget(self.llm_endpoint_test_status)

        self.scroll_container.add(self.llm_endpoint_panel)
        self.sync_llm_provider_panels()

        button_layout = QHBoxLayout()
        button_layout.setSpacing(theme.config.spacing.medium)
        save_btn = PrimaryButton(text="Save", command=partial(self.on_save_section_clicked, section_name))
        button_layout.addWidget(save_btn)
        reset_btn = DangerButton(
            text="Reset to Defaults",
            command=partial(self.on_reset_section_clicked, section_name),
        )
        button_layout.addWidget(reset_btn)
        button_layout.addStretch()
        self.scroll_container.content_layout.addLayout(button_layout)
        self.scroll_container.content_layout.addSpacing(theme.config.spacing.large)

        self.section_widgets[section_name] = {"widgets": llm_widgets}
        self.sync_llm_model_ui_state()

    def sync_llm_provider_panels(self) -> None:
        combo = self.setting_widgets.get("llm.provider")
        if not isinstance(combo, QComboBox):
            return
        provider = combo.currentData() or "shipped"
        shipped = provider == "shipped"
        self.llm_shipped_panel.setVisible(shipped)
        self.llm_endpoint_panel.setVisible(not shipped)

    def on_llm_provider_changed(self, _index: int) -> None:
        self.sync_llm_provider_panels()

    def on_llm_endpoint_test_clicked(self) -> None:
        if not self.controller:
            return
        updates: dict = {"llm.provider": "local_endpoint"}
        for setting_key in ("llm.local_endpoint_base_url", "llm.local_endpoint_model"):
            widget = self.setting_widgets.get(setting_key)
            if isinstance(widget, TextInput):
                updates[setting_key] = widget.text().strip()
        self.llm_endpoint_test_btn.setEnabled(False)
        asyncio.create_task(self._run_endpoint_test(updates))

    async def _run_endpoint_test(self, updates: dict) -> None:
        ok, msg = await self.controller.update_settings_async(updates)
        if ok:
            ok, msg = await self.controller.test_local_endpoint_async()
        elif not msg:
            msg = "Could not save endpoint settings."
        self.endpoint_test_result.emit(ok, msg)

    @Slot(bool, str)
    def on_llm_endpoint_test_finished(self, ok: bool, message: str) -> None:
        btn = getattr(self, "llm_endpoint_test_btn", None)
        if btn:
            btn.setEnabled(True)
        label = getattr(self, "llm_endpoint_test_status", None)
        if label is None:
            return
        label.setText(message)
        color = theme.config.text.light if ok else theme.config.shapes.orange
        pal = label.palette()
        pal.setColor(QPalette.ColorRole.WindowText, QColor(color))
        label.setPalette(pal)

    def apply_llm_model_combo_style(self, combo: QComboBox) -> None:
        combo.setFont(theme.get_font(size="medium"))

    def set_llm_download_busy(self, busy: bool) -> None:
        self.llm_download_busy = busy
        btn = getattr(self, "llm_download_btn", None)
        if btn:
            btn.setEnabled(not busy)

    def set_llm_status_label_muted(self, lbl: SmallLabel) -> None:
        pal = lbl.palette()
        pal.setColor(QPalette.ColorRole.WindowText, QColor(theme.config.text.medium))
        lbl.setPalette(pal)

    def sync_llm_model_ui_state(self) -> None:
        if not self.controller:
            return
        combo = self.setting_widgets.get("llm.selected_model_id")
        lbl = getattr(self, "llm_availability_label", None)
        btn = getattr(self, "llm_download_btn", None)
        if not isinstance(combo, QComboBox) or lbl is None or btn is None:
            return
        raw = combo.currentData()
        combo_mid = str(raw) if raw else DEFAULT_LLM_MODEL_ID
        llm_settings = self.settings.get("llm") if isinstance(self.settings.get("llm"), dict) else {}
        eff = str(llm_settings.get("selected_model_id") or DEFAULT_LLM_MODEL_ID)
        eff_ok = self.controller.llm_bundle_on_disk(eff)
        combo_ok = self.controller.llm_bundle_on_disk(combo_mid)

        if self.llm_download_busy:
            self.set_llm_status_label_muted(lbl)
            lbl.setText("Download in progress…")
            lbl.setVisible(True)
            btn.setVisible(True)
            return

        if eff_ok and combo_ok:
            lbl.clear()
            lbl.setVisible(False)
            btn.setVisible(False)
            return

        self.set_llm_status_label_muted(lbl)
        lbl.setVisible(True)
        btn.setVisible(True)
        if not eff_ok:
            lbl.setText("The saved model is not fully on this device. Download it, or choose another installed model.")
        else:
            lbl.setText("This model is not on this device. Download to use it; until then the saved model stays active.")

    def on_llm_model_combo_changed(self, _index: int) -> None:
        if getattr(self, "suppress_llm_combo_events", False) or not self.controller:
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
        self.sync_llm_model_ui_state()

    def on_llm_download_clicked(self) -> None:
        if not self.controller or self.llm_download_busy:
            return
        combo = self.setting_widgets.get("llm.selected_model_id")
        if not isinstance(combo, QComboBox):
            return
        model_id = combo.currentData()
        if not model_id:
            return
        mid = str(model_id)
        if self.controller.llm_bundle_on_disk(mid):
            self.sync_llm_model_ui_state()
            return

        spec = get_whitelisted_llm_model(mid)
        dlg = LlmDownloadProgressDialog(self, model_label=spec.label if spec else mid)
        dlg.cancel_clicked.connect(self.controller.cancel_llm_download)

        self.llm_active_download = {"dlg": dlg, "mid": mid, "download_msg": ""}
        self.set_llm_download_busy(True)
        self.sync_llm_model_ui_state()

        self.controller.llm_download_progress.connect(dlg.set_status)
        self.controller.llm_cancellable_download_finished.connect(self.on_llm_cancellable_download_finished)
        self.controller.llm_download_cancelled.connect(self.on_llm_download_cancelled)
        self.controller.llm_download_integrity_error.connect(self.on_llm_download_integrity_error)
        try:
            self.controller.schedule_llm_cancellable_download(mid)
            dlg.exec()
        finally:
            for sig, slot in (
                (
                    self.controller.llm_cancellable_download_finished,
                    self.on_llm_cancellable_download_finished,
                ),
                (
                    self.controller.llm_download_cancelled,
                    self.on_llm_download_cancelled,
                ),
                (
                    self.controller.llm_download_integrity_error,
                    self.on_llm_download_integrity_error,
                ),
            ):
                try:
                    sig.disconnect(slot)
                except TypeError:
                    pass

    def _close_download_dialog(self) -> None:
        ctx = getattr(self, "llm_active_download", None)
        if not ctx:
            return
        dlg = ctx["dlg"]
        try:
            self.controller.llm_download_progress.disconnect(dlg.set_status)
        except TypeError:
            pass

    def on_llm_cancellable_download_finished(self, ok: bool, msg: str) -> None:
        ctx = getattr(self, "llm_active_download", None)
        if not ctx or not self.controller:
            self.logger.warning("LLM download finished without active context (ok=%s)", ok)
            return
        dlg = ctx["dlg"]
        mid = ctx["mid"]
        self._close_download_dialog()
        self.logger.info("LLM download finished ok=%s mid=%s", ok, mid)

        if ok:
            ctx["download_msg"] = msg

            async def persist() -> Tuple[bool, str]:
                return await self.controller.update_setting_async("llm.selected_model_id", mid)

            fut = asyncio.create_task(persist())
            fut.add_done_callback(self.on_llm_persist_future_done)
        else:
            dlg.apply_outcome(False, msg)
            self.llm_active_download = None
            self.set_llm_download_busy(False)
            self.sync_llm_model_ui_state()
            QMessageBox.warning(self, "Download failed", msg)

    def on_llm_download_cancelled(self) -> None:
        ctx = getattr(self, "llm_active_download", None)
        if not ctx:
            return
        self._close_download_dialog()
        ctx["dlg"].apply_outcome(False, "")
        self.llm_active_download = None
        self.set_llm_download_busy(False)
        self.sync_llm_model_ui_state()

    def on_llm_download_integrity_error(self, msg: str) -> None:
        ctx = getattr(self, "llm_active_download", None)
        if not ctx:
            return
        self._close_download_dialog()
        ctx["dlg"].apply_outcome(False, msg)
        self.llm_active_download = None
        self.set_llm_download_busy(False)
        self.sync_llm_model_ui_state()
        QMessageBox.critical(self, "Integrity verification failed", msg)

    def on_llm_persist_future_done(self, fut) -> None:
        s_ok, s_msg = fut.result()
        self.llm_persist_finished.emit(s_ok, s_msg)

    @Slot(bool, str)
    def on_llm_persist_finished_main(self, s_ok: bool, s_msg: str) -> None:
        ctx = getattr(self, "llm_active_download", None)
        if not ctx or not self.controller:
            self.set_llm_download_busy(False)
            self.sync_llm_model_ui_state()
            return
        dlg = ctx["dlg"]
        download_msg = ctx.get("download_msg", "")
        self.llm_active_download = None

        dlg.apply_outcome(s_ok, s_msg if not s_ok else download_msg)
        self.set_llm_download_busy(False)
        self.sync_llm_model_ui_state()
        if s_ok:
            self.controller.load_settings()
        else:
            QMessageBox.warning(self, "Could not activate model", s_msg)
        self.logger.info("LLM download persisted and UI cleared (s_ok=%s)", s_ok)

    def on_save_section_clicked(self, section_name: str) -> None:
        """Handle save button clicked for a specific section."""
        if not self.controller:
            return

        section_info = self.section_widgets.get(section_name)
        if not section_info:
            self.logger.warning("Section not found: %s", section_name)
            return

        settings_to_save = {}
        for setting_key, widget in section_info["widgets"].items():
            if setting_key in ("llm.selected_model_id",):
                continue
            if setting_key == "llm.provider" and isinstance(widget, QComboBox):
                value = widget.currentData()
                if value is None:
                    self.show_error(f"Invalid value for {setting_key}")
                    return
                settings_to_save[setting_key] = str(value)
                continue
            if isinstance(widget, Checkbox):
                settings_to_save[setting_key] = widget.isChecked()
            elif isinstance(widget, QComboBox):
                value = widget.currentData()
                if value is None:
                    self.show_error(f"Invalid value for {setting_key}")
                    return
                settings_to_save[setting_key] = str(value)
            elif isinstance(widget, TextInput):
                field = FIELD_BY_PATH.get(setting_key)
                if field is None:
                    self.show_error(f"Unknown setting: {setting_key}")
                    return
                text_value = widget.text().strip()
                try:
                    value = field.value_type(text_value)
                except (ValueError, TypeError):
                    self.show_error(f"Invalid value for '{field.label}'. Expected {field.value_type.__name__}.")
                    return
                min_val, max_val = get_config_field_bounds(setting_key)
                if (min_val is not None and value < min_val) or (max_val is not None and value > max_val):
                    self.show_error(f"'{field.label}' must be between {min_val} and {max_val}.")
                    return
                settings_to_save[setting_key] = value

        if settings_to_save:
            self.pending_save_section = section_name
            self.controller.update_settings(settings_to_save)
            self.logger.info(
                "Saved %s settings from section: %s",
                len(settings_to_save),
                section_name,
            )

    def on_reset_section_clicked(self, section_name: str) -> None:
        """Handle reset button clicked for a specific section."""
        reply = QMessageBox.question(
            self,
            "Reset Section",
            f"Reset {section_name} to defaults?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            if not self.controller:
                return
            section_info = self.section_widgets.get(section_name)
            if not section_info:
                return
            setting_keys = list(section_info["widgets"].keys())
            self.controller.reset_section_settings(setting_keys)
            self.logger.info("Reset section to defaults: %s", section_name)
            QMessageBox.information(self, "Success", f"{section_name} reset to defaults!")

    def show_error(self, message: str) -> None:
        """Show error message dialog."""
        QMessageBox.critical(self, "Error", message)
