"""Attention Mechanism - production-grade cognitive focus and resource allocation."""

from __future__ import annotations

import atexit
import threading
import time
import uuid
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional
import weakref

from ...core.config import AttentionConfig
from ...utils.logging import setup_logging
from ...utils.validators import InputValidator
from .attention_lifecycle import AttentionLifecycleMixin
from .exceptions import AttentionError, CapacityExceededError, InvalidStimulus
from .models import AttentionItem, AttentionMetrics


logger = setup_logging(level="INFO", include_module=True)


class AttentionMechanism(AttentionLifecycleMixin):
    """
    Production-grade attention mechanism for cognitive resource allocation
    
    This class implements a biologically-inspired attention system with:
    - Thread-safe concurrent access
    - Comprehensive error handling and validation
    - Performance monitoring and metrics
    - Configurable parameters and behavior
    - Memory-efficient resource management
    - Security-conscious input sanitization
    
    The mechanism follows Miller's rule of 7±2 items in active attention,
    models cognitive fatigue and recovery, and provides adaptive focus
    management based on stimulus salience and task priority.
    """
    
    # Shared thread pool for async operations to avoid race conditions and bottlenecks
    _shared_executor: Optional[ThreadPoolExecutor] = None
    _executor_lock = threading.Lock()

    def __init__(
        self,
        config: Optional[AttentionConfig] = None,
        validator: Optional[InputValidator] = None,
        max_history_size: int = 1000,
        enable_metrics: bool = True,
        cleanup_interval_seconds: float = 300.0
    ):
        """
        Initialize production-grade attention mechanism
        
        Args:
            config: Attention configuration object
            validator: Input validator instance
            max_history_size: Maximum size of attention history
            enable_metrics: Whether to collect performance metrics
            cleanup_interval_seconds: Interval for automatic cleanup
            
        Raises:
            AttentionError: If initialization parameters are invalid
        """
        try:
            # Configuration and validation
            self.config = config or AttentionConfig()
            self.validator = validator or InputValidator()
            
            # Validate configuration
            self._validate_config()
            
            # Thread safety
            self._lock = threading.RLock()
            
            # Core attention state (thread-safe)
            self.focused_items: Dict[str, AttentionItem] = {}
            self.attention_history: deque = deque(maxlen=max_history_size)
            
            # Fatigue and resource management
            self.current_fatigue = 0.0
            self.total_cognitive_load = 0.0
            self.attention_capacity = 1.0
            
            # Performance tracking
            self.last_update = datetime.now()
            self.attention_episodes = deque(maxlen=500)
            self.metrics = AttentionMetrics() if enable_metrics else None
            
            # Resource management
            self.cleanup_interval = cleanup_interval_seconds
            self._last_cleanup = datetime.now()
            self._is_shutdown = False
            
            # Weak references for memory management
            self._observers: List[weakref.ReferenceType] = []
            
            # Processing statistics
            self._processing_times = deque(maxlen=100)
            self._error_counts = defaultdict(int)
            
            logger.info(f"Attention mechanism initialized with config: "
                       f"max_items={self.config.max_attention_items}, "
                       f"threshold={self.config.salience_threshold:.3f}, "
                       f"fatigue_rate={self.config.fatigue_decay_rate:.3f}")
                       
        except Exception as e:
            logger.error(f"Failed to initialize attention mechanism: {e}")
            raise AttentionError(f"Initialization failed: {e}") from e
    
    def _validate_config(self) -> None:
        """Validate attention configuration parameters"""
        if not isinstance(self.config, AttentionConfig):
            raise AttentionError("config must be an AttentionConfig instance")
        
        validators = [
            (self.config.max_attention_items, 1, 20, "max_attention_items"),
            (self.config.salience_threshold, 0.0, 1.0, "salience_threshold"),
            (self.config.fatigue_decay_rate, 0.0, 1.0, "fatigue_decay_rate"),
            (self.config.attention_recovery_rate, 0.0, 1.0, "attention_recovery_rate")
        ]
        
        for value, min_val, max_val, name in validators:
            if not isinstance(value, (int, float)):
                raise AttentionError(f"{name} must be numeric")
            if not (min_val <= value <= max_val):
                raise AttentionError(f"{name} must be between {min_val} and {max_val}")

    def _validate_allocation_params(
        self,
        stimulus_id: str,
        content: Any,
        salience: float,
        novelty: float,
        priority: float,
        effort_required: float
    ) -> Dict[str, Any]:
        """
        Validate and sanitize attention allocation parameters
        
        Returns:
            Dictionary of validated parameters
            
        Raises:
            InvalidStimulus: If any parameter is invalid
        """
        # Validate stimulus ID
        if not stimulus_id or not isinstance(stimulus_id, str):
            raise InvalidStimulus("stimulus_id must be a non-empty string")
        
        if len(stimulus_id) > 200:  # Security: prevent excessively long IDs
            raise InvalidStimulus("stimulus_id too long (max 200 characters)")
        
        # Validate numeric parameters
        numeric_params = {
            'salience': salience,
            'novelty': novelty, 
            'priority': priority,
            'effort_required': effort_required
        }
        
        for param_name, value in numeric_params.items():
            if not isinstance(value, (int, float)):
                raise InvalidStimulus(f"{param_name} must be numeric, got {type(value)}")
            if not (0.0 <= value <= 1.0):
                raise InvalidStimulus(f"{param_name} must be between 0.0 and 1.0, got {value}")
        
        # Sanitize content if it's a string
        sanitized_content = content
        if isinstance(content, str):
            # Basic sanitization - remove control characters, limit length
            sanitized_content = ''.join(char for char in content if ord(char) >= 32)
            if len(sanitized_content) > 10000:  # Security: prevent excessively large content
                sanitized_content = sanitized_content[:10000] + "...[truncated]"
        
        return {
            'stimulus_id': stimulus_id,
            'content': sanitized_content,
            'salience': float(salience),
            'novelty': float(novelty),
            'priority': float(priority),
            'effort_required': float(effort_required)
        }
    
    def _create_allocation_result(
        self,
        allocated: bool,
        reason: Optional[str] = None,
        attention_score: Optional[float] = None,
        effective_salience: Optional[float] = None,
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create standardized attention allocation result
        
        Args:
            allocated: Whether attention was successfully allocated
            reason: Reason for allocation failure (if any)
            attention_score: Calculated attention score
            effective_salience: Effective salience after novelty boost
            correlation_id: Request correlation ID
            
        Returns:
            Standardized allocation result dictionary
        """
        result = {
            "allocated": allocated,
            "timestamp": datetime.now().isoformat(),
            "cognitive_load": self.total_cognitive_load,
            "fatigue_level": self.current_fatigue,
            "focused_items": len(self.focused_items),
            "attention_capacity": self.attention_capacity
        }
        
        if reason:
            result["reason"] = reason
        
        if attention_score is not None:
            result["attention_score"] = attention_score
        
        if effective_salience is not None:
            result["effective_salience"] = effective_salience
        
        if correlation_id:
            result["correlation_id"] = correlation_id
        
        return result
    
    def _is_capacity_exceeded(self, additional_effort: float) -> bool:
        """Check if adding item would exceed cognitive capacity"""
        projected_load = self.total_cognitive_load + additional_effort
        return (len(self.focused_items) >= self.config.max_attention_items or 
                projected_load > self.attention_capacity)
    
    def _manage_capacity(self, new_priority: float, new_effort: float) -> bool:
        """
        Manage attention capacity by removing low-priority items
        
        Returns:
            True if capacity was successfully managed, False otherwise
        """
        if not self.focused_items:
            return True
        
        # Sort by priority and activation (lowest first)
        sorted_items = sorted(
            self.focused_items.items(),
            key=lambda x: (x[1].priority, x[1].activation)
        )
        
        initial_count = len(self.focused_items)
        
        # Remove lowest priority items until there's room
        for item_id, item in sorted_items:
            if (item.priority < new_priority and 
                len(self.focused_items) > 0):
                self._remove_from_focus(item_id)
                logger.debug(f"Removed {item_id} from focus for capacity management")
                
                # Check if we have enough capacity now
                if not self._is_capacity_exceeded(new_effort):
                    return True
        
        # If we still don't have capacity after trying to remove items
        return len(self.focused_items) < initial_count
    
    def _update_cognitive_state(self, effort_required: float):
        """Update cognitive load and fatigue"""
        # Increase fatigue based on effort
        fatigue_increase = effort_required * self.config.fatigue_decay_rate * 0.1
        self.current_fatigue = min(1.0, self.current_fatigue + fatigue_increase)

        # Penalty for recent switches (attention residue modeling)
        if self.metrics and self.metrics.focus_switches > 0:
            self.current_fatigue = min(1.0, self.current_fatigue + 0.01)  # minor attention residue

        # Recalculate total cognitive load
        self._recalculate_cognitive_load()
    
    def _calculate_attention_score(
        self, 
        salience: float, 
        priority: float, 
        effort_required: float
    ) -> float:
        """
        Calculate overall attention score with enhanced weighting
        
        Args:
            salience: Stimulus salience (0.0 to 1.0)
            priority: Task priority (0.0 to 1.0)
            effort_required: Cognitive effort needed (0.0 to 1.0)
            
        Returns:
            Calculated attention score (0.0 to 1.0)
        """
        # Base score from salience and priority
        base_score = (salience * 0.6) + (priority * 0.4)
        
        # Adjust for current cognitive state
        fatigue_penalty = self.current_fatigue * 0.3
        load_penalty = (self.total_cognitive_load / self.attention_capacity) * 0.2
        
        # Effort consideration (harder tasks get less attention when fatigued)
        effort_penalty = effort_required * fatigue_penalty * 0.5
        
        # Apply penalties
        final_score = base_score - fatigue_penalty - load_penalty - effort_penalty
        
        # Ensure score stays within bounds
        return max(0.0, min(1.0, final_score))
    
    def _recalculate_cognitive_load(self):
        """Recalculate total cognitive load from current focus items"""
        total_load = 0.0
        for item in self.focused_items.values():
            # Load is effort weighted by activation
            item_load = item.effort_required * item.activation
            total_load += item_load
        
        self.total_cognitive_load = total_load
    
    def _remove_from_focus(self, item_id: str):
        """
        Remove item from attention focus with proper cleanup and logging
        
        Args:
            item_id: ID of item to remove
        """
        if item_id not in self.focused_items:
            logger.warning(f"Attempted to remove non-existent item: {item_id}")
            return
        
        try:
            item = self.focused_items[item_id]
            
            # Create attention episode record
            episode = {
                "item_id": item_id,
                "start_time": item.created_at.isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_seconds": item.duration_seconds,
                "final_activation": item.activation,
                "salience": item.salience,
                "priority": item.priority,
                "correlation_id": item.correlation_id
            }
            self.attention_episodes.append(episode)
            
            # Remove from focused items
            del self.focused_items[item_id]
            
            # Notify observers
            self._notify_observers("focus_lost", {
                "item_id": item_id,
                "episode": episode
            })
            
            logger.debug(f"Removed {item_id} from attention focus "
                        f"(duration: {item.duration_seconds:.1f}s, "
                        f"final_activation: {item.activation:.3f})",
                        extra={"correlation_id": item.correlation_id})
                        
        except Exception as e:
            logger.error(f"Error removing item {item_id} from focus: {e}")
            # Force removal to prevent stuck items
            if item_id in self.focused_items:
                del self.focused_items[item_id]
    
    def _log_attention_event(self, stimulus_id: str, salience: float, attention_score: float):
        """
        Log attention allocation event with comprehensive metadata
        
        Args:
            stimulus_id: ID of the stimulus
            salience: Effective salience score
            attention_score: Calculated attention score
        """
        try:
            event = {
                "timestamp": datetime.now().isoformat(),
                "stimulus_id": stimulus_id,
                "salience": salience,
                "attention_score": attention_score,
                "cognitive_load": self.total_cognitive_load,
                "fatigue_level": self.current_fatigue,
                "items_in_focus": len(self.focused_items),
                "available_capacity": max(0.0, self.attention_capacity - self.total_cognitive_load)
            }
            self.attention_history.append(event)
            
            # Notify observers
            self._notify_observers("attention_allocated", event)
            
        except Exception as e:
            logger.error(f"Error logging attention event: {e}")

    @contextmanager
    def _timed_operation(self, operation_name: str) -> Iterator[None]:
        """Context manager for timing operations and updating metrics"""
        start_time = time.time()
        correlation_id = str(uuid.uuid4())
        
        try:
            logger.debug(f"Starting {operation_name}", extra={"correlation_id": correlation_id})
            yield
            
        except Exception as e:
            self._error_counts[operation_name] += 1
            if self.metrics:
                self.metrics.error_count += 1
            logger.error(f"Error in {operation_name}: {e}", 
                        extra={"correlation_id": correlation_id})
            raise
            
        finally:
            elapsed = time.time() - start_time
            self._processing_times.append(elapsed)
            
            if self.metrics:
                self.metrics.total_processing_time += elapsed
            
            logger.debug(f"Completed {operation_name} in {elapsed:.3f}s",
                        extra={"correlation_id": correlation_id})
    
    def allocate_attention(
        self,
        stimulus_id: str,
        content: Any,
        salience: float,
        novelty: float = 0.0,
        priority: float = 0.5,
        effort_required: float = 0.5
    ) -> Dict[str, Any]:
        """
        Thread-safe attention allocation with comprehensive validation
        (Refactored: broken into smaller methods for clarity and maintainability)
        """
        if self._is_shutdown:
            raise AttentionError("Attention mechanism is shutdown")

        with self._timed_operation("allocate_attention"):
            if self.metrics:
                self.metrics.total_allocations += 1
            try:
                with self._lock:
                    validated_params, effective_salience, dynamic_threshold = self._validate_and_prepare_allocation(
                        stimulus_id, content, salience, novelty, priority, effort_required
                    )
                    if effective_salience < dynamic_threshold:
                        logger.debug(f"Stimulus {stimulus_id} below dynamic threshold "
                                     f"({effective_salience:.3f} < {dynamic_threshold:.3f})")
                        return self._create_allocation_result(
                            allocated=False,
                            reason="below_dynamic_threshold",
                            effective_salience=effective_salience
                        )
                    self._check_capacity_and_manage(validated_params)
                    attention_score, correlation_id, is_new = self._create_or_update_attention_item(
                        stimulus_id, validated_params, effective_salience
                    )
                    self._finalize_allocation(
                        stimulus_id, validated_params, effective_salience, attention_score, correlation_id, is_new
                    )
                    return self._create_allocation_result(
                        allocated=True,
                        attention_score=attention_score,
                        effective_salience=effective_salience,
                        correlation_id=correlation_id
                    )
            except (InvalidStimulus, CapacityExceededError):
                raise
            except Exception as e:
                logger.error(f"Unexpected error in attention allocation: {e}")
                raise AttentionError(f"Allocation failed: {e}") from e

    def _validate_and_prepare_allocation(
        self,
        stimulus_id: str,
        content: Any,
        salience: float,
        novelty: float,
        priority: float,
        effort_required: float
    ) -> tuple:
        """
        Validate and prepare allocation parameters, calculate effective salience and threshold.
        Caller must hold self._lock.
        Returns (validated_params, effective_salience, dynamic_threshold)
        """
        validated_params = self._validate_allocation_params(
            stimulus_id, content, salience, novelty, priority, effort_required
        )
        self._auto_cleanup()
        novelty_boost = getattr(self.config, 'novelty_boost', 0.2)
        if hasattr(self.config, 'allow_dynamic_novelty') and self.config.allow_dynamic_novelty:
            novelty_boost = validated_params['novelty']
        effective_salience = min(1.0, validated_params['salience'] + (validated_params['novelty'] * novelty_boost))
        dynamic_threshold = self.config.salience_threshold + (self.current_fatigue * 0.1)
        return validated_params, effective_salience, dynamic_threshold

    def _check_capacity_and_manage(self, validated_params: dict) -> None:
        """
        Check and manage attention capacity, raise if cannot allocate.
        Caller must hold self._lock.
        """
        if self._is_capacity_exceeded(validated_params['effort_required']):
            managed = self._manage_capacity(
                validated_params['priority'],
                validated_params['effort_required']
            )
            if not managed:
                if self.metrics:
                    self.metrics.capacity_exceeded_count += 1
                raise CapacityExceededError(
                    "Cannot allocate attention: capacity exceeded"
                )

    def _create_or_update_attention_item(
        self,
        stimulus_id: str,
        validated_params: dict,
        effective_salience: float
    ) -> tuple:
        """
        Create or update the attention item, return (attention_score, correlation_id, is_new)
        Caller must hold self._lock.
        """
        attention_score = self._calculate_attention_score(
            effective_salience,
            validated_params['priority'],
            validated_params['effort_required']
        )
        correlation_id = str(uuid.uuid4())
        is_new = False
        if stimulus_id in self.focused_items:
            item = self.focused_items[stimulus_id]
            old_activation = item.activation
            item.update_activation(attention_score * 0.1)
            item.priority = max(item.priority, validated_params['priority'])
            logger.debug(f"Updated existing item {stimulus_id}: "
                         f"activation {old_activation:.3f} -> {item.activation:.3f}",
                         extra={"correlation_id": correlation_id})
        else:
            item = AttentionItem(
                id=stimulus_id,
                content=validated_params['content'],
                salience=effective_salience,
                activation=attention_score,
                created_at=datetime.now(),
                last_updated=datetime.now(),
                priority=validated_params['priority'],
                effort_required=validated_params['effort_required'],
                correlation_id=correlation_id
            )
            self.focused_items[stimulus_id] = item
            is_new = True
            if self.metrics:
                self.metrics.focus_switches += 1
            logger.info(f"Added new attention item {stimulus_id} "
                        f"(salience: {effective_salience:.3f}, score: {attention_score:.3f})",
                        extra={"correlation_id": correlation_id})
        return attention_score, correlation_id, is_new

    def _finalize_allocation(
        self,
        stimulus_id: str,
        validated_params: dict,
        effective_salience: float,
        attention_score: float,
        correlation_id: str,
        is_new: bool
    ) -> None:
        """
        Finalize allocation: update cognitive state, log event, update metrics.
        Caller must hold self._lock.
        """
        self._update_cognitive_state(validated_params['effort_required'])
        self._log_attention_event(stimulus_id, effective_salience, attention_score)
        if self.metrics:
            self.metrics.successful_allocations += 1
            self.metrics.average_attention_score = (
                (self.metrics.average_attention_score * (self.metrics.successful_allocations - 1) +
                 attention_score) / self.metrics.successful_allocations
            )
            self.metrics.peak_cognitive_load = max(
                self.metrics.peak_cognitive_load, self.total_cognitive_load
            )

atexit.register(AttentionMechanism.shutdown_shared_executor)

__all__ = [
    "AttentionError",
    "AttentionItem",
    "AttentionMechanism",
    "AttentionMetrics",
    "CapacityExceededError",
    "InvalidStimulus",
]
