import asyncio
import logging
import uuid
from typing import Any, Dict, Optional, Tuple

from PySide6.QtCore import Signal

from vocalance.app.config.app_config import GlobalAppConfig, get_whitelisted_llm_model
from vocalance.app.event_bus import EventBus
from vocalance.app.events.core_events import (
    LlmUiNotificationEvent,
    LlmUiRequestEvent,
    RuntimeConfigRequestEvent,
    RuntimeConfigResponseEvent,
    SettingsChangedEvent,
)
from vocalance.app.ui.controls.qt_base_controller import QtBaseController
from vocalance.app.utils.llm_dep_check import llm_deps_available


class QtSettingsController(QtBaseController):
    settings_loaded = Signal(dict)
    setting_changed = Signal(str, object)
    all_settings_changed = Signal(dict)
    operation_error = Signal(str)
    llm_bundle_status_updated = Signal()
    llm_download_progress = Signal(str)
    llm_cancellable_download_finished = Signal(bool, str)
    llm_download_cancelled = Signal()
    llm_download_integrity_error = Signal(str)

    def __init__(self, event_bus: EventBus, config: GlobalAppConfig) -> None:
        super().__init__(event_bus=event_bus, logger=logging.getLogger("QtSettingsController"))
        self.config = config
        self._llm_enabled = llm_deps_available()
        self.cached_settings: Dict[str, Any] = {}
        self.llm_bundle_status: Dict[str, bool] = {}
        self.pending_setting_futures: Dict[str, asyncio.Future[Tuple[bool, str]]] = {}
        self.active_llm_download_rid: Optional[str] = None
        self.subscribe(SettingsChangedEvent, self.on_settings_changed)
        self.subscribe(RuntimeConfigResponseEvent, self.on_runtime_config_response)
        if self._llm_enabled:
            self.subscribe(LlmUiNotificationEvent, self.on_llm_ui_notification)

    def on_settings_changed(self, settings_change: SettingsChangedEvent) -> None:
        self.cached_settings = settings_change.all_settings
        self.all_settings_changed.emit(self.cached_settings)
        for key, value in settings_change.updated_settings.items():
            self.setting_changed.emit(key, value)

    def on_runtime_config_response(self, event: RuntimeConfigResponseEvent) -> None:
        if event.op == "effective_snapshot":
            self.cached_settings = event.all_settings
            self.settings_loaded.emit(self.cached_settings)
            if self._llm_enabled:
                asyncio.create_task(self.event_bus.publish(LlmUiRequestEvent(op="refresh_bundle_status")))
            return
        if event.op == "update_result":
            fut = self.pending_setting_futures.pop(event.correlation_id, None)
            if fut is not None and not fut.done():
                fut.set_result((event.success, event.message))
            if not event.success and event.message:
                self.operation_error.emit(event.message)

    def on_llm_ui_notification(self, event: LlmUiNotificationEvent) -> None:
        if event.kind == "bundle_status":
            self.llm_bundle_status = dict(event.status)
            self.llm_bundle_status_updated.emit()
        elif event.kind == "download_progress":
            if self.active_llm_download_rid and event.request_id == self.active_llm_download_rid:
                self.llm_download_progress.emit(event.message)
        elif event.kind == "download_finished":
            if self.active_llm_download_rid and event.request_id == self.active_llm_download_rid:
                self.llm_cancellable_download_finished.emit(event.ok, event.message)
                self.active_llm_download_rid = None
            asyncio.create_task(self.event_bus.publish(LlmUiRequestEvent(op="refresh_bundle_status")))
        elif event.kind == "download_cancelled":
            if self.active_llm_download_rid and event.request_id == self.active_llm_download_rid:
                self.llm_download_cancelled.emit()
                self.active_llm_download_rid = None
            asyncio.create_task(self.event_bus.publish(LlmUiRequestEvent(op="refresh_bundle_status")))
        elif event.kind == "download_integrity_error":
            if self.active_llm_download_rid and event.request_id == self.active_llm_download_rid:
                self.llm_download_integrity_error.emit(event.message)
                self.active_llm_download_rid = None
            asyncio.create_task(self.event_bus.publish(LlmUiRequestEvent(op="refresh_bundle_status")))

    def load_settings(self) -> None:
        asyncio.create_task(self.event_bus.publish(RuntimeConfigRequestEvent(op="get_effective")))

    async def update_setting_async(self, key: str, value: Any) -> Tuple[bool, str]:
        cid = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[Tuple[bool, str]] = loop.create_future()
        self.pending_setting_futures[cid] = fut
        await self.event_bus.publish(
            RuntimeConfigRequestEvent(op="update", updates={key: value}, correlation_id=cid),
        )
        return await fut

    def update_setting(self, key: str, value: Any) -> None:
        asyncio.create_task(self.update_setting_task(key, value))

    async def update_setting_task(self, key: str, value: Any) -> None:
        ok, msg = await self.update_setting_async(key, value)
        if ok:
            parts = key.split(".")
            if len(parts) == 2:
                category, setting_key = parts
                self.cached_settings.setdefault(category, {})[setting_key] = value
        elif msg:
            self.operation_error.emit(msg)

    def llm_bundle_on_disk(self, model_id: str) -> bool:
        if not get_whitelisted_llm_model(model_id):
            return False
        return bool(self.llm_bundle_status.get(model_id))

    def schedule_llm_cancellable_download(self, model_id: str) -> str:
        rid = str(uuid.uuid4())
        self.active_llm_download_rid = rid
        asyncio.create_task(self.event_bus.publish(LlmUiRequestEvent(op="start_download", model_id=model_id, request_id=rid)))
        return rid

    def cancel_llm_download(self) -> None:
        rid = self.active_llm_download_rid or ""
        asyncio.create_task(self.event_bus.publish(LlmUiRequestEvent(op="cancel_download", request_id=rid)))

    async def update_settings_async(self, settings: Dict[str, Any]) -> Tuple[bool, str]:
        cid = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[Tuple[bool, str]] = loop.create_future()
        self.pending_setting_futures[cid] = fut
        await self.event_bus.publish(RuntimeConfigRequestEvent(op="update", updates=dict(settings), correlation_id=cid))
        ok, msg = await fut
        if ok:
            for key, value in settings.items():
                parts = key.split(".")
                if len(parts) == 2:
                    category, setting_key = parts
                    self.cached_settings.setdefault(category, {})[setting_key] = value
            self.all_settings_changed.emit(self.cached_settings)
        elif msg:
            self.operation_error.emit(msg)
        return ok, msg

    def update_settings(self, settings: Dict[str, Any]) -> None:
        asyncio.create_task(self.update_settings_async(settings))

    def reset_to_defaults(self) -> None:
        asyncio.create_task(self.event_bus.publish(RuntimeConfigRequestEvent(op="reset_defaults")))

    def reset_section_settings(self, setting_keys: list) -> None:
        asyncio.create_task(self.reset_section_task(setting_keys))

    async def reset_section_task(self, setting_keys: list) -> None:
        await self.event_bus.publish(RuntimeConfigRequestEvent(op="reset_section", setting_keys=tuple(setting_keys)))
        await self.event_bus.publish(RuntimeConfigRequestEvent(op="get_effective"))

    def get_setting(self, key: str, default: Any = None) -> Any:
        return self.cached_settings.get(key, default)

    def get_all_settings(self) -> Dict[str, Any]:
        return dict(self.cached_settings)
