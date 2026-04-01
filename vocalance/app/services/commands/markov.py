"""Markov-chain next-command prediction from history (disabled while dictation is active)."""

from __future__ import annotations

import logging
import time
from collections import Counter, defaultdict, deque
from typing import Deque, Dict, List, Optional, Tuple

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.event_bus import EventBus
from vocalance.app.events.core_events import AudioDetectedEvent, MarkovPredictionEvent, MarkovPredictionFeedbackEvent
from vocalance.app.events.dictation_events import DictationModeDisableOthersEvent
from vocalance.app.services.storage.storage_models import CommandHistoryData, CommandHistoryEntry
from vocalance.app.services.storage.storage_service import StorageService

logger = logging.getLogger(__name__)


class MarkovCommandService:
    def __init__(self, event_bus: EventBus, config: GlobalAppConfig, storage: StorageService) -> None:
        self._event_bus = event_bus
        self._config = config
        self._markov_config = config.markov_predictor
        self._storage = storage
        self._transition_counts: Dict[int, Dict[tuple, Counter]] = {
            2: defaultdict(Counter),
            3: defaultdict(Counter),
            4: defaultdict(Counter),
        }
        self._command_history: Deque[str] = deque(maxlen=self._markov_config.max_order)
        self._model_trained = False
        self._last_prediction_time = 0.0
        self._prediction_cooldown = self._markov_config.prediction_cooldown_seconds
        self._pending_prediction: Optional[Tuple[str, float]] = None
        self._cooldown_remaining = 0
        self._dictation_active = False

    async def initialize(self) -> bool:
        try:
            await self._train_model()
            await self._seed_command_history()
            return True
        except Exception as e:
            logger.error("Failed to initialize predictor: %s", e, exc_info=True)
            return False

    def setup_subscriptions(self) -> None:
        self._event_bus.subscribe(event_type=AudioDetectedEvent, handler=self._handle_audio_detected_fast_track)
        self._event_bus.subscribe(event_type=MarkovPredictionFeedbackEvent, handler=self._handle_prediction_feedback)
        self._event_bus.subscribe(event_type=DictationModeDisableOthersEvent, handler=self._handle_dictation_mode_change)

    async def _train_model(self) -> None:
        for order in range(self._markov_config.min_order, self._markov_config.max_order + 1):
            self._transition_counts[order].clear()

        for order in range(self._markov_config.min_order, self._markov_config.max_order + 1):
            await self._train_order(order)

        self._model_trained = True

    async def _seed_command_history(self) -> None:
        try:
            history_data = await self._storage.read(model_type=CommandHistoryData)
        except Exception as e:
            logger.error("Failed to seed command history: %s", e)
            return

        if not history_data or not history_data.history:
            return

        recent = [entry.command for entry in history_data.history[-self._markov_config.max_order :]]
        for cmd in recent:
            self._command_history.append(cmd)

    async def _train_order(self, order: int) -> None:
        history = await self._load_filtered_history(order)
        if len(history) < order + 1:
            return

        commands = [cmd.command for cmd in history]
        for i in range(len(commands) - order):
            ctx = tuple(commands[i : i + order])
            nxt = commands[i + order]
            self._transition_counts[order][ctx][nxt] += 1

    async def _load_filtered_history(self, order: int) -> List[CommandHistoryEntry]:
        history_data = await self._storage.read(model_type=CommandHistoryData)
        all_history = history_data.history
        if not all_history:
            return []

        days_window = self._markov_config.training_window_days.get(order, 7)
        commands_window = self._markov_config.training_window_commands.get(order, 1000)
        cutoff = time.time() - (days_window * 86400)
        filtered = [cmd for cmd in all_history if cmd.timestamp >= cutoff]
        if len(filtered) > commands_window:
            filtered = filtered[-commands_window:]
        return filtered

    async def _handle_dictation_mode_change(self, event: DictationModeDisableOthersEvent) -> None:
        prev = self._dictation_active
        self._dictation_active = event.dictation_mode_active
        if prev != self._dictation_active:
            logger.info(
                "Markov predictor %s",
                "DISABLED (dictation active)" if self._dictation_active else "ENABLED",
            )

    async def _handle_audio_detected_fast_track(self, event: AudioDetectedEvent) -> None:
        try:
            now = time.time()
            if now - self._last_prediction_time < self._prediction_cooldown:
                return
            if not self._markov_config.enabled or self._dictation_active or self._cooldown_remaining > 0:
                return
            if not self._model_trained or len(self._command_history) < self._markov_config.min_order:
                return

            prediction = self._predict_next_command()
            if not prediction:
                return

            predicted_cmd, confidence, order_used = prediction
            if confidence < self._markov_config.confidence_threshold:
                return

            self._last_prediction_time = now
            self._pending_prediction = (predicted_cmd, confidence)
            logger.info("Markov prediction (order-%s): %r (confidence=%.2f%%)", order_used, predicted_cmd, confidence * 100)

            await self._event_bus.publish(
                MarkovPredictionEvent(
                    predicted_command=predicted_cmd,
                    confidence=confidence,
                    audio_id=int(now * 1_000_000),
                )
            )
        except Exception as e:
            logger.error("Error in Markov prediction handler: %s", e, exc_info=True)

    async def _handle_prediction_feedback(self, event: MarkovPredictionFeedbackEvent) -> None:
        actual = event.actual_command
        if event.predicted_command != actual and event.was_correct is False:
            logger.warning(
                "Markov prediction incorrect: predicted %r, actual %r — cooldown",
                event.predicted_command,
                actual,
            )
            self._cooldown_remaining = self._markov_config.incorrect_prediction_cooldown
        elif event.predicted_command == actual and event.was_correct:
            logger.info("Markov prediction correct: %r", event.predicted_command)

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
        self._command_history.append(actual)

    def _predict_next_command(self) -> Optional[Tuple[str, float, int]]:
        for order in range(self._markov_config.max_order, self._markov_config.min_order - 1, -1):
            if len(self._command_history) < order:
                continue
            ctx = tuple(list(self._command_history)[-order:])
            transitions = self._transition_counts[order].get(ctx)
            if not transitions:
                continue

            total = sum(transitions.values())
            min_freq = self._markov_config.min_command_frequency.get(order, 2)
            valid = {cmd: c for cmd, c in transitions.items() if c >= min_freq}
            if not valid:
                continue

            predicted_cmd, count = max(valid.items(), key=lambda x: x[1])
            return (predicted_cmd, count / total, order)
        return None

    async def retrain(self) -> bool:
        try:
            await self._train_model()
            return True
        except Exception as e:
            logger.error("Error during retraining: %s", e, exc_info=True)
            return False

    def on_enabled_updated(self, enabled: bool) -> None:
        self._markov_config.enabled = enabled

    def on_confidence_threshold_updated(self, threshold: float) -> None:
        self._markov_config.confidence_threshold = threshold

    async def shutdown(self) -> None:
        logger.debug("Markov predictor shutdown")
