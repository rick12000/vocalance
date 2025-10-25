import logging
import time
from collections import Counter, defaultdict, deque
from typing import Dict, List, Optional, Tuple

from vocalance.app.config.app_config import GlobalAppConfig
from vocalance.app.event_bus import EventBus
from vocalance.app.events.core_events import AudioDetectedEvent, MarkovPredictionEvent, MarkovPredictionFeedbackEvent
from vocalance.app.services.storage.storage_models import CommandHistoryData, CommandHistoryEntry
from vocalance.app.services.storage.storage_service import StorageService

logger = logging.getLogger(__name__)


class MarkovCommandService:
    """Multi-order Markov predictor with backoff strategy for command prediction.

    Trains 2nd through 4th order Markov chains on command history, uses backoff from
    highest to lowest order for prediction, and provides feedback-based cooldown on
    incorrect predictions to maintain accuracy.
    """

    def __init__(self, event_bus: EventBus, config: GlobalAppConfig, storage: StorageService) -> None:
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

        logger.debug(f"MarkovCommandService initialized (orders {self._markov_config.min_order}-{self._markov_config.max_order})")

    async def initialize(self) -> bool:
        """Initialize predictor by training models on stored command history.

        Returns:
            True if initialization succeeded, False otherwise.
        """
        try:
            await self._train_model()
            return True
        except Exception as e:
            logger.error(f"Failed to initialize predictor: {e}", exc_info=True)
            return False

    def setup_subscriptions(self) -> None:
        """Setup event subscriptions for audio detection and prediction feedback."""
        self._event_bus.subscribe(event_type=AudioDetectedEvent, handler=self._handle_audio_detected_fast_track)
        self._event_bus.subscribe(event_type=MarkovPredictionFeedbackEvent, handler=self._handle_prediction_feedback)

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

    async def _handle_audio_detected_fast_track(self, event: AudioDetectedEvent) -> None:
        """Predict and execute command immediately when audio detected for ultra-low latency.

        Uses backoff Markov prediction based on recent command history to execute the
        most likely next command before STT completes, subject to cooldown and confidence.

        Args:
            event: Audio detection event from recorder
        """
        try:
            current_time = time.time()

            # Cooldown to prevent spam
            if current_time - self._last_prediction_time < self._prediction_cooldown:
                return

            if not self._markov_config.enabled:
                return

            # Check if we're in cooldown from incorrect prediction
            if self._cooldown_remaining > 0:
                logger.debug(f"Skipping Markov (cooldown: {self._cooldown_remaining} commands remaining)")
                return

            if not self._model_trained or len(self._command_history) < self._markov_config.min_order:
                return

            # Backoff prediction using context
            prediction = self._predict_next_command()

            if prediction:
                predicted_cmd, confidence, order = prediction

                if confidence >= self._markov_config.confidence_threshold:
                    self._last_prediction_time = current_time

                    # Store prediction for feedback verification
                    self._pending_prediction = (predicted_cmd, confidence)

                    logger.info(f"ULTRA-FAST (order-{order}): '{predicted_cmd}' (confidence={confidence:.2%})")

                    await self._event_bus.publish(
                        MarkovPredictionEvent(
                            predicted_command=predicted_cmd, confidence=confidence, audio_id=int(current_time * 1000000)
                        )
                    )

        except Exception as e:
            logger.error(f"Error in fast-track handling: {e}", exc_info=True)

    async def _handle_prediction_feedback(self, event: MarkovPredictionFeedbackEvent) -> None:
        """Process prediction accuracy feedback and manage cooldown.

        Updates in-memory command history with actual executed command and enters
        cooldown mode on incorrect predictions to maintain accuracy.

        Args:
            event: Feedback event with prediction accuracy and actual command
        """
        try:
            actual_command = event.actual_command
            was_correct = event.was_correct
            source = event.source

            if was_correct:
                logger.info(f"Markov prediction CORRECT (verified by {source}): '{event.predicted_command}'")
            else:
                logger.warning(
                    f"Markov prediction INCORRECT (verified by {source}): "
                    f"predicted '{event.predicted_command}', actual '{actual_command}' - entering cooldown"
                )
                # Enter cooldown mode
                self._cooldown_remaining = self._markov_config.incorrect_prediction_cooldown

            # Update in-memory command history (for model predictions)
            self._command_history.append(actual_command)

        except Exception as e:
            logger.error(f"Error handling prediction feedback: {e}", exc_info=True)

    def _predict_next_command(self) -> Optional[Tuple[str, float, int]]:
        """Predict next command using backoff from highest to lowest order.

        Attempts prediction starting at max_order, falling back to lower orders if
        no valid transitions found, returns first match above minimum frequency.

        Returns:
            Tuple of (predicted_command, confidence, order_used) or None if no prediction
        """
        for order in range(self._markov_config.max_order, self._markov_config.min_order - 1, -1):
            if len(self._command_history) < order:
                continue

            # Get context of length 'order'
            context = tuple(list(self._command_history)[-order:])

            if context not in self._transition_counts[order]:
                continue

            transitions = self._transition_counts[order][context]

            if not transitions:
                continue

            total_count = sum(transitions.values())

            # Get order-specific minimum frequency
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
        """Manually trigger model retraining"""
        try:
            await self._train_model()
            return True
        except Exception as e:
            logger.error(f"Error during retraining: {e}", exc_info=True)
            return False

    def on_confidence_threshold_updated(self, threshold: float) -> None:
        """Handle real-time confidence threshold update from settings.

        Args:
            threshold: New confidence threshold value (0.0 to 1.0)
        """
        old_threshold = self._markov_config.confidence_threshold
        self._markov_config.confidence_threshold = threshold
        logger.info(f"Markov predictor confidence threshold updated: {old_threshold:.2f} -> {threshold:.2f}")

    async def shutdown(self) -> None:
        try:
            logger.info("Markov predictor shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)
