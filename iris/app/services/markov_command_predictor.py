"""
Markov Chain Command Predictor

Predicts next command based on historical command sequences.
Uses first-order Markov chain to enable ultra-low latency command execution.
"""

import logging
import time
import asyncio
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter, deque
from datetime import datetime, timedelta

from iris.app.event_bus import EventBus
from iris.app.config.app_config import GlobalAppConfig
from iris.app.services.storage.storage_adapters import CommandHistoryStorageAdapter
from iris.app.events.core_events import CommandAudioSegmentReadyEvent, AudioDetectedEvent
from iris.app.events.markov_events import MarkovPredictionEvent, CommandExecutedEvent
from iris.app.events.stt_events import CommandTextRecognizedEvent
from iris.app.events.command_events import (
    AutomationCommandParsedEvent,
    MarkCommandParsedEvent,
    GridCommandParsedEvent,
    DictationCommandParsedEvent
)

logger = logging.getLogger(__name__)


class MarkovCommandPredictor:
    """Backoff Markov chain predictor (2nd-4th order) for command sequences"""
    
    def __init__(
        self,
        event_bus: EventBus,
        config: GlobalAppConfig,
        history_adapter: CommandHistoryStorageAdapter
    ):
        self._event_bus = event_bus
        self._config = config
        self._markov_config = config.markov_predictor
        self._history_adapter = history_adapter
        
        # Multi-order transition counts: {order: {context: Counter}}
        self._transition_counts: Dict[int, Dict[tuple, Counter]] = {
            2: defaultdict(Counter),
            3: defaultdict(Counter),
            4: defaultdict(Counter)
        }
        
        # Command history buffer for context
        self._command_history: deque = deque(maxlen=self._markov_config.max_order)
        self._model_trained = False
        
        # Batch writes for performance
        self._pending_commands: deque = deque(maxlen=100)
        self._write_task: Optional[asyncio.Task] = None
        self._write_interval = 5.0
        
        # Fast-track prediction on audio detection
        self._last_prediction_time = 0.0
        self._prediction_cooldown = 0.05
        
        # Prediction verification and feedback
        self._pending_prediction: Optional[Tuple[str, float]] = None
        self._cooldown_remaining = 0
        
        logger.info(f"MarkovCommandPredictor initialized (orders {self._markov_config.min_order}-{self._markov_config.max_order})")
    
    async def initialize(self) -> bool:
        """Initialize and train the model"""
        try:
            await self._train_model()
            # Start batch write task
            self._write_task = asyncio.create_task(self._batch_write_loop())
            return True
        except Exception as e:
            logger.error(f"Failed to initialize predictor: {e}", exc_info=True)
            return False
    
    def setup_subscriptions(self) -> None:
        """Setup event subscriptions"""
        self._event_bus.subscribe(AudioDetectedEvent, self._handle_audio_detected_fast_track)
        self._event_bus.subscribe(CommandTextRecognizedEvent, self._handle_stt_feedback)
        self._event_bus.subscribe(AutomationCommandParsedEvent, self._handle_command_executed)
        self._event_bus.subscribe(MarkCommandParsedEvent, self._handle_command_executed)
        self._event_bus.subscribe(GridCommandParsedEvent, self._handle_command_executed)
        self._event_bus.subscribe(DictationCommandParsedEvent, self._handle_command_executed)
        
        logger.info("Markov predictor event subscriptions configured with fast-track and feedback")
    
    async def _train_model(self) -> None:
        """Train multi-order Markov models on historical command data (async)"""
        try:
            # Clear existing models
            for order in range(self._markov_config.min_order, self._markov_config.max_order + 1):
                self._transition_counts[order].clear()
            
            # Train each order separately
            for order in range(self._markov_config.min_order, self._markov_config.max_order + 1):
                await self._train_order(order)
            
            self._model_trained = True
            
        except Exception as e:
            logger.error(f"Error training model: {e}", exc_info=True)
    
    async def _train_order(self, order: int) -> None:
        """Train a specific order Markov chain"""
        try:
            history = await self._load_filtered_history(order)
            
            if len(history) < order + 1:
                logger.info(f"Insufficient history for order-{order} chain (need {order + 1}, have {len(history)})")
                return
            
            # Extract command texts
            commands = [cmd["command"] for cmd in history]
            
            # Build transitions for this order
            transitions_built = 0
            for i in range(len(commands) - order):
                context = tuple(commands[i:i + order])
                next_cmd = commands[i + order]
                self._transition_counts[order][context][next_cmd] += 1
                transitions_built += 1
            
            logger.info(
                f"Order-{order} chain trained: {len(history)} commands, "
                f"{transitions_built} transitions, "
                f"{len(self._transition_counts[order])} unique contexts"
            )
            
        except Exception as e:
            logger.error(f"Error training order-{order} model: {e}", exc_info=True)
    
    async def _load_filtered_history(self, order: int) -> List[Dict]:
        """Load and filter command history based on config for specific order"""
        all_history = await self._history_adapter.load_history()
        
        if not all_history:
            return []
        
        # Get order-specific windows
        days_window = self._markov_config.training_window_days.get(order, 7)
        commands_window = self._markov_config.training_window_commands.get(order, 1000)
        
        cutoff_timestamp = time.time() - (days_window * 86400)
        filtered = [
            cmd for cmd in all_history
            if cmd.get("timestamp", 0) >= cutoff_timestamp
        ]
        
        if len(filtered) > commands_window:
            filtered = filtered[-commands_window:]
        
        return filtered
    
    async def _handle_audio_detected_fast_track(self, event: AudioDetectedEvent) -> None:
        """FAST-TRACK: Predict immediately when audio is first detected"""
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
                    
                    logger.info(
                        f"⚡ ULTRA-FAST (order-{order}): '{predicted_cmd}' (confidence={confidence:.2%})"
                    )
                    
                    await self._event_bus.publish(
                        MarkovPredictionEvent(
                            predicted_command=predicted_cmd,
                            confidence=confidence,
                            audio_id=int(current_time * 1000000)
                        )
                    )
        
        except Exception as e:
            logger.error(f"Error in fast-track handling: {e}", exc_info=True)
    
    async def _handle_stt_feedback(self, event: CommandTextRecognizedEvent) -> None:
        """Handle STT feedback to verify Markov predictions"""
        try:
            # Only process Vosk feedback (true STT), not Markov predictions
            if event.engine != "vosk":
                return
            
            actual_command = event.text.strip().lower()
            
            # Check if we have a pending prediction to verify
            if self._pending_prediction:
                predicted_cmd, confidence = self._pending_prediction
                self._pending_prediction = None
                
                if actual_command == predicted_cmd:
                    # Correct prediction!
                    logger.info(f"✓ Markov prediction CORRECT: '{predicted_cmd}'")
                    
                    # Add to command history (verified by STT)
                    timestamp = time.time()
                    self._pending_commands.append({
                        "command": actual_command,
                        "timestamp": timestamp
                    })
                    
                    # Update in-memory model
                    self._command_history.append(actual_command)
                    
                    # Publish verified command event
                    await self._event_bus.publish(
                        CommandExecutedEvent(
                            command_text=actual_command,
                            timestamp=timestamp
                        )
                    )
                    
                else:
                    # Incorrect prediction!
                    logger.warning(
                        f"✗ Markov prediction INCORRECT: predicted '{predicted_cmd}', "
                        f"actual '{actual_command}' - entering cooldown"
                    )
                    
                    # Enter cooldown mode
                    self._cooldown_remaining = self._markov_config.incorrect_prediction_cooldown
                    
                    # Add actual command to history
                    timestamp = time.time()
                    self._pending_commands.append({
                        "command": actual_command,
                        "timestamp": timestamp
                    })
                    
                    self._command_history.append(actual_command)
                    
                    await self._event_bus.publish(
                        CommandExecutedEvent(
                            command_text=actual_command,
                            timestamp=timestamp
                        )
                    )
            
        except Exception as e:
            logger.error(f"Error handling STT feedback: {e}", exc_info=True)
    
    def _predict_next_command(self) -> Optional[Tuple[str, float, int]]:
        """Predict next command using backoff strategy (tries highest order first)"""
        
        # Try from highest to lowest order
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
            valid_transitions = {
                cmd: count for cmd, count in transitions.items()
                if count >= min_freq
            }
            
            if not valid_transitions:
                continue
            
            # Found valid prediction at this order
            most_common_cmd = max(valid_transitions.items(), key=lambda x: x[1])
            predicted_cmd, count = most_common_cmd
            
            confidence = count / total_count
            
            return (predicted_cmd, confidence, order)
        
        return None
    
    async def _handle_command_executed(self, event) -> None:
        """Track executed commands ONLY when NOT from Markov prediction"""
        try:
            command_text = self._extract_command_text(event)
            
            if not command_text:
                return
            
            # Decrement cooldown counter
            if self._cooldown_remaining > 0:
                self._cooldown_remaining -= 1
                logger.debug(f"Cooldown decremented: {self._cooldown_remaining} remaining")
            
            # Only add to history if not already added by feedback mechanism
            # (feedback mechanism adds verified Markov predictions)
            if not self._pending_prediction:
                timestamp = time.time()
                
                # Add to pending queue
                self._pending_commands.append({
                    "command": command_text,
                    "timestamp": timestamp
                })
                
                # Update command history buffer
                self._command_history.append(command_text)
                
                # Publish event
                await self._event_bus.publish(
                    CommandExecutedEvent(
                        command_text=command_text,
                        timestamp=timestamp
                    )
                )
                
                logger.debug(f"Command tracked: '{command_text}'")
            
        except Exception as e:
            logger.error(f"Error tracking command: {e}", exc_info=True)
    
    async def _batch_write_loop(self) -> None:
        """Background task that batches writes to storage"""
        while True:
            try:
                await asyncio.sleep(self._write_interval)
                
                if self._pending_commands:
                    # Get all pending commands
                    commands_to_write = list(self._pending_commands)
                    self._pending_commands.clear()
                    
                    # Load current history
                    history = await self._history_adapter.load_history()
                    
                    # Append new commands
                    history.extend(commands_to_write)
                    
                    # Write in one batch
                    await self._history_adapter.save_history(history)
                    
                    logger.debug(f"Batch wrote {len(commands_to_write)} commands to storage")
            
            except asyncio.CancelledError:
                # Flush remaining commands before exiting
                if self._pending_commands:
                    try:
                        commands_to_write = list(self._pending_commands)
                        history = await self._history_adapter.load_history()
                        history.extend(commands_to_write)
                        await self._history_adapter.save_history(history)
                        logger.info(f"Flushed {len(commands_to_write)} commands on shutdown")
                    except Exception as e:
                        logger.error(f"Error flushing commands: {e}")
                break
            
            except Exception as e:
                logger.error(f"Error in batch write loop: {e}", exc_info=True)
    
    def _extract_command_text(self, event) -> Optional[str]:
        """Extract command text from various command events"""
        if isinstance(event, DictationCommandParsedEvent):
            return None
        
        if hasattr(event, 'command') and hasattr(event.command, 'command_key'):
            return event.command.command_key
        
        if hasattr(event, 'command_text'):
            return event.command_text
        
        return None
    
    async def retrain(self) -> bool:
        """Manually trigger model retraining"""
        try:
            await self._train_model()
            return True
        except Exception as e:
            logger.error(f"Error during retraining: {e}", exc_info=True)
            return False
    
    async def shutdown(self) -> None:
        """Shutdown predictor and flush pending writes"""
        try:
            if self._write_task:
                self._write_task.cancel()
                try:
                    await self._write_task
                except asyncio.CancelledError:
                    pass
            logger.info("Markov predictor shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)
    
    def get_status(self) -> Dict:
        """Get predictor status"""
        return {
            "enabled": self._markov_config.enabled,
            "model_trained": self._model_trained,
            "transitions_by_order": {
                order: len(counts) for order, counts in self._transition_counts.items()
            },
            "command_history_size": len(self._command_history),
            "confidence_threshold": self._markov_config.confidence_threshold,
            "pending_writes": len(self._pending_commands),
            "cooldown_remaining": self._cooldown_remaining,
            "pending_prediction": self._pending_prediction is not None
        }

