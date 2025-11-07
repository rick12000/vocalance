import logging
import time
from collections import Counter, defaultdict, deque
from typing import Dict, List, Optional, Tuple

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.event_bus import EventBus
from vocalance.app.events.core_events import AudioDetectedEvent, MarkovPredictionEvent, MarkovPredictionFeedbackEvent
from vocalance.app.events.dictation_events import DictationModeDisableOthersEvent
from vocalance.app.services.storage.storage_models import CommandHistoryData, CommandHistoryEntry
from vocalance.app.services.storage.storage_service import StorageService

logger = logging.getLogger(__name__)


class MarkovCommandService:
    """Multi-order Markov predictor with backoff strategy for command prediction.

    Trains 2nd through 4th order Markov chains on command history, uses backoff from
    highest to lowest order for prediction, and provides feedback-based cooldown on
    incorrect predictions to maintain accuracy.

    CRITICAL: Predictor is automatically disabled during dictation mode to prevent
    premature "stop dictation" predictions from interrupting active dictation sessions.
    """

    def __init__(self, event_bus: EventBus, config: GlobalAppConfig, storage: StorageService) -> None:
        """Initialize Markov predictor with configuration.

        Args:
            event_bus: EventBus for pub/sub messaging.
            config: Global application configuration.
            storage: Storage service for command history access.
        """
        self._event_bus: EventBus = event_bus
        self._config: GlobalAppConfig = config
        self._markov_config = config.markov_predictor
        self._storage: StorageService = storage
        self._transition_counts: Dict[int, Dict[tuple, Counter]] = {
            2: defaultdict(Counter),
            3: defaultdict(Counter),
            4: defaultdict(Counter),
        }
        self._command_history: deque = deque(maxlen=self._markov_config.max_order)
        self._model_trained: bool = False
        self._last_prediction_time: float = 0.0
        self._prediction_cooldown: float = self._markov_config.prediction_cooldown_seconds
        self._pending_prediction: Optional[Tuple[str, float]] = None
        self._cooldown_remaining: int = 0
        self._dictation_active: bool = False  # Track dictation state to prevent predictions during dictation

        logger.debug(f"MarkovCommandService initialized (orders {self._markov_config.min_order}-{self._markov_config.max_order})")

    async def initialize(self) -> bool:
        """Initialize predictor by training models on stored command history.

        Returns:
            True if initialization succeeded, False otherwise.
        """
        try:
            await self._train_model()
            await self._seed_command_history()
            return True
        except Exception as e:
            logger.error(f"Failed to initialize predictor: {e}", exc_info=True)
            return False

    def setup_subscriptions(self) -> None:
        """Setup event subscriptions for audio detection, prediction feedback, and dictation state."""
        self._event_bus.subscribe(event_type=AudioDetectedEvent, handler=self._handle_audio_detected_fast_track)
        self._event_bus.subscribe(event_type=MarkovPredictionFeedbackEvent, handler=self._handle_prediction_feedback)
        self._event_bus.subscribe(event_type=DictationModeDisableOthersEvent, handler=self._handle_dictation_mode_change)

        logger.debug("Markov predictor event subscriptions configured")

    async def _train_model(self) -> None:
        """Train all Markov chain orders on filtered historical command data.

        Loads command history from storage, filters by configured time and count
        windows for each order, and builds transition count matrices for prediction.
        """
        try:
            for order in range(self._markov_config.min_order, self._markov_config.max_order + 1):
                self._transition_counts[order].clear()

            for order in range(self._markov_config.min_order, self._markov_config.max_order + 1):
                await self._train_order(order)

            self._model_trained = True

        except Exception as e:
            logger.error(f"Error training model: {e}", exc_info=True)

    async def _seed_command_history(self) -> None:
        """Seed in-memory command history with recent commands for immediate prediction capability.

        Loads the most recent max_order commands from storage to populate the prediction
        context window, enabling predictions to work immediately after startup instead of
        requiring feedback events to build up the history first.
        """
        try:
            history_data = await self._storage.read(model_type=CommandHistoryData)
            if history_data and history_data.history:
                recent_commands = [entry.command for entry in history_data.history[-self._markov_config.max_order :]]

                for cmd in recent_commands:
                    self._command_history.append(cmd)

                logger.info(f"Seeded command history with {len(recent_commands)} recent commands")
            else:
                logger.debug("No command history available to seed predictor context")
        except Exception as e:
            logger.error(f"Failed to seed command history (predictions will build up over time): {e}")

    async def _train_order(self, order: int) -> None:
        """Train a specific order Markov chain.

        Args:
            order: Order of the Markov chain to train.
        """
        try:
            history = await self._load_filtered_history(order)

            if len(history) < order + 1:
                logger.debug(f"Insufficient history for order-{order} chain (need {order + 1}, have {len(history)})")
                return

            commands = [cmd.command for cmd in history]

            transitions_built = 0
            for i in range(len(commands) - order):
                context = tuple(commands[i : i + order])
                next_cmd = commands[i + order]
                self._transition_counts[order][context][next_cmd] += 1
                transitions_built += 1

            logger.debug(
                f"Order-{order} chain trained: {len(history)} commands, "
                f"{transitions_built} transitions, "
                f"{len(self._transition_counts[order])} unique contexts"
            )

        except Exception as e:
            logger.error(f"Error training order-{order} model: {e}", exc_info=True)

    async def _load_filtered_history(self, order: int) -> List[CommandHistoryEntry]:
        """Load and filter command history based on config for specific Markov order.

        Applies time window and command count filters configured for the specified order
        to ensure training data is appropriately scoped.

        Args:
            order: Markov chain order for which to load filtered history.

        Returns:
            List of filtered CommandHistoryEntry objects for training.
        """
        history_data = await self._storage.read(model_type=CommandHistoryData)
        all_history = history_data.history

        if not all_history:
            return []

        # Get order-specific windows
        days_window = self._markov_config.training_window_days.get(order, 7)
        commands_window = self._markov_config.training_window_commands.get(order, 1000)

        cutoff_timestamp = time.time() - (days_window * 86400)
        filtered = [cmd for cmd in all_history if cmd.timestamp >= cutoff_timestamp]

        if len(filtered) > commands_window:
            filtered = filtered[-commands_window:]

        return filtered

    async def _handle_dictation_mode_change(self, event: DictationModeDisableOthersEvent) -> None:
        """Handle dictation mode changes to prevent predictions during active dictation.

        CRITICAL: This prevents the predictor from firing during dictation, which would
        cause premature "stop dictation" predictions after seeing the activation word,
        interrupting the actual dictation before any text is captured.

        Args:
            event: Dictation mode change event
        """
        try:
            old_state = self._dictation_active
            self._dictation_active = event.dictation_mode_active

            if old_state != self._dictation_active:
                if self._dictation_active:
                    logger.info("Markov predictor DISABLED - dictation mode active (prevents premature stop predictions)")
                else:
                    logger.info("Markov predictor ENABLED - dictation mode inactive")
        except Exception as e:
            logger.error(f"Error handling dictation mode change: {e}", exc_info=True)

    async def _handle_audio_detected_fast_track(self, event: AudioDetectedEvent) -> None:
        """Predict and execute command immediately when audio detected for ultra-low latency.

        Uses backoff Markov prediction based on recent command history to execute the
        most likely next command before STT completes, subject to cooldown and confidence.

        CRITICAL: Automatically disabled during dictation mode to prevent premature
        "stop dictation" predictions from interrupting active dictation sessions.

        Args:
            event: Audio detection event from recorder.
        """
        try:
            current_time = time.time()
            time_since_last = current_time - self._last_prediction_time

            # Time-based cooldown to prevent prediction spam
            if time_since_last < self._prediction_cooldown:
                return

            # Check if predictor is enabled and conditions are met
            if not self._markov_config.enabled or self._dictation_active or self._cooldown_remaining > 0:
                return

            # Ensure model is trained and sufficient history exists
            if not self._model_trained or len(self._command_history) < self._markov_config.min_order:
                return

            # Attempt backoff prediction using command history context
            prediction = self._predict_next_command()

            if prediction:
                predicted_cmd, confidence, order = prediction

                if confidence >= self._markov_config.confidence_threshold:
                    self._last_prediction_time = current_time
                    self._pending_prediction = (predicted_cmd, confidence)

                    logger.info(f"Markov prediction (order-{order}): '{predicted_cmd}' (confidence={confidence:.2%})")

                    await self._event_bus.publish(
                        MarkovPredictionEvent(
                            predicted_command=predicted_cmd, confidence=confidence, audio_id=int(current_time * 1000000)
                        )
                    )

        except Exception as e:
            logger.error(f"Error in Markov prediction handler: {e}", exc_info=True)

    async def _handle_prediction_feedback(self, event: MarkovPredictionFeedbackEvent) -> None:
        """Process prediction accuracy feedback and manage cooldown.

        Updates in-memory command history with actual executed command and enters
        cooldown mode on incorrect predictions to maintain accuracy. Decrements
        cooldown counter on each command execution.

        Args:
            event: Feedback event with prediction accuracy and actual command.
        """
        try:
            actual_command = event.actual_command
            was_correct = event.was_correct

            # Handle incorrect predictions by entering cooldown mode
            if event.predicted_command != actual_command and was_correct is False:
                logger.warning(
                    f"Markov prediction incorrect: predicted '{event.predicted_command}', "
                    f"actual '{actual_command}' - entering cooldown"
                )
                self._cooldown_remaining = self._markov_config.incorrect_prediction_cooldown
            elif event.predicted_command == actual_command and was_correct:
                logger.info(f"Markov prediction correct: '{event.predicted_command}'")

            # Decrement cooldown on every command execution (if currently in cooldown)
            if self._cooldown_remaining > 0:
                self._cooldown_remaining -= 1

            # Update in-memory command history with actual command for future predictions
            self._command_history.append(actual_command)

        except Exception as e:
            logger.error(f"Error handling prediction feedback: {e}", exc_info=True)

    def _predict_next_command(self) -> Optional[Tuple[str, float, int]]:
        """Predict next command using backoff from highest to lowest order.

        Attempts prediction starting at max_order, falling back to lower orders if
        no valid transitions found. Returns first match above minimum frequency threshold.

        Returns:
            Tuple of (predicted_command, confidence, order_used) or None if no prediction available.
        """
        for order in range(self._markov_config.max_order, self._markov_config.min_order - 1, -1):
            if len(self._command_history) < order:
                continue

            context = tuple(list(self._command_history)[-order:])

            if context not in self._transition_counts[order]:
                continue

            transitions = self._transition_counts[order][context]

            if not transitions:
                continue

            total_count = sum(transitions.values())
            min_freq = self._markov_config.min_command_frequency.get(order, 2)
            valid_transitions = {cmd: count for cmd, count in transitions.items() if count >= min_freq}

            if not valid_transitions:
                continue

            # Found valid prediction at this order
            most_common_cmd = max(valid_transitions.items(), key=lambda x: x[1])
            predicted_cmd, count = most_common_cmd
            confidence = count / total_count

            return (predicted_cmd, confidence, order)

        return None

    async def retrain(self) -> bool:
        """Manually trigger model retraining.

        Returns:
            True if retraining succeeded, False otherwise.
        """
        try:
            await self._train_model()
            return True
        except Exception as e:
            logger.error(f"Error during retraining: {e}", exc_info=True)
            return False

    def on_enabled_updated(self, enabled: bool) -> None:
        """Handle real-time enabled setting update from settings.

        Args:
            enabled: New enabled state.
        """
        self._markov_config.enabled = enabled
        logger.info(f"Markov predictor {'enabled' if enabled else 'disabled'}")

    def on_confidence_threshold_updated(self, threshold: float) -> None:
        """Handle real-time confidence threshold update from settings.

        Args:
            threshold: New confidence threshold value (0.0 to 1.0).
        """
        self._markov_config.confidence_threshold = threshold
        logger.info(f"Markov confidence threshold updated to {threshold:.2%}")

    async def shutdown(self) -> None:
        """Shutdown predictor and cleanup resources."""
        try:
            logger.info("Markov predictor shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)
