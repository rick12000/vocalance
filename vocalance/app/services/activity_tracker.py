from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import IO, Optional

from vocalance.app.config.app_config import ActivityTrackingConfig
from vocalance.app.config.command_types import AutomationCommand, ParameterizedCommand


class ActivityTracker:
    """Writes structured JSON activity events to a per-session JSONL log file.

    Tracks two classes of events when enabled:
    - Final dictation outputs (standard, streaming, and LLM-enhanced modes).
    - PyAutoGUI automation executions with full upstream command context.

    All writes are no-ops when ``enabled`` is False, preserving privacy by
    default. Thread-safe: can be called from both the asyncio event loop and
    blocking worker threads.

    Attributes:
        enabled: Whether activity tracking is active.
        run_id: UUID4 stable for the lifetime of this tracker instance.
        activity_logs_dir: Directory where the JSONL log file is written.
        log_filename: Filename for this session's log, e.g. ``activity_20260618_143300.jsonl``.
        lock: Threading lock used to serialise file writes.
        log_file: Open file handle, created lazily on first write.
    """

    def __init__(self, config: ActivityTrackingConfig, activity_logs_dir: str) -> None:
        self.enabled = config.enabled
        self.run_id = str(uuid.uuid4())
        self.activity_logs_dir = activity_logs_dir
        self.log_filename = f"activity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        self.lock = threading.Lock()
        self.log_file: Optional[IO[str]] = None

    def get_log_file(self) -> Optional[IO[str]]:
        """Return the open log file handle, creating it lazily on first call.

        Returns:
            Open file handle in append mode, or None when tracking is disabled.
        """
        if not self.enabled:
            return None
        with self.lock:
            if self.log_file is None:
                path = os.path.join(self.activity_logs_dir, self.log_filename)
                self.log_file = open(path, "a", encoding="utf-8")
        return self.log_file

    def write_event(self, payload: dict) -> None:
        """Serialise and append one JSON record to the activity log.

        Args:
            payload: Event-specific dict merged with ubiquitous fields before writing.
        """
        f = self.get_log_file()
        if f is None:
            return
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            **payload,
        }
        with self.lock:
            f.write(json.dumps(record) + "\n")
            f.flush()

    def log_dictation(
        self,
        text: str,
        mode: str,
        session_id: str,
        llm_enhanced: bool,
        active_modifiers: set[str],
    ) -> None:
        """Record a final dictation output event.

        Should be called after the text has been successfully typed into the
        target application.

        Args:
            text: The final processed text that was typed into the target application.
            mode: Dictation mode string (e.g. "standard", "smart", "visual").
            session_id: UUID of the dictation session that produced this output.
            llm_enhanced: True when the text was processed by the LLM before output.
            active_modifiers: Set of active modifier IDs applied during the session.
        """
        self.write_event(
            {
                "event_type": "dictation",
                "dictation": {
                    "text": text,
                    "mode": mode,
                    "session_id": session_id,
                    "llm_enhanced": llm_enhanced,
                    "active_modifiers": sorted(active_modifiers),
                },
            }
        )

    def log_automation(self, command: AutomationCommand, count: int) -> None:
        """Record a PyAutoGUI automation execution event.

        Should be called after the action has been executed via pyautogui.
        Captures the full upstream command context including the trigger phrase,
        action details, and whether the command is custom or built-in.

        Args:
            command: The resolved automation command that was executed.
            count: Number of times the action was repeated.
        """
        self.write_event(
            {
                "event_type": "automation",
                "automation": {
                    "command_key": command.command_key,
                    "action_type": command.action_type,
                    "action_value": command.action_value,
                    "count": count,
                    "is_custom": command.is_custom,
                    "functional_group": command.functional_group,
                    "short_description": command.short_description,
                    "command_variant": "parameterized" if isinstance(command, ParameterizedCommand) else "exact_match",
                },
            }
        )

    async def shutdown(self) -> None:
        """Flush and close the activity log file."""
        with self.lock:
            if self.log_file is not None:
                self.log_file.close()
                self.log_file = None
