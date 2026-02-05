"""
Attention Mechanism - Production-grade cognitive focus and resource allocation

This module implements a biologically-inspired attention mechanism that manages
cognitive focus, resource allocation, and fatigue modeling with thread safety,
comprehensive error handling, and performance monitoring.

Features:
- Thread-safe attention allocation and management
- Comprehensive input validation and error handling
- Performance monitoring and metrics collection
- Configurable parameters with production defaults
- Structured logging with correlation IDs
- Memory-efficient resource management
- Security-conscious parameter sanitization
"""

import threading
import time
import uuid
import asyncio
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Iterator
from concurrent.futures import ThreadPoolExecutor
import weakref

from ...core.config import AttentionConfig
from ...utils.validators import InputValidator
from ...utils.logging import setup_logging

# Configure structured logging
logger = setup_logging(level="INFO", include_module=True)

class AttentionError(Exception):
    """Base exception for attention mechanism errors"""
    pass


class CapacityExceededError(AttentionError):
    """Raised when attention capacity is exceeded"""
    pass


class InvalidStimulus(AttentionError):
    """Raised when stimulus parameters are invalid"""
    pass


@dataclass
class AttentionMetrics:
    """Performance and usage metrics for attention mechanism"""
    total_allocations: int = 0
    successful_allocations: int = 0
    capacity_exceeded_count: int = 0
    focus_switches: int = 0
    average_attention_score: float = 0.0
    peak_cognitive_load: float = 0.0
    total_processing_time: float = 0.0
    error_count: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate allocation success rate"""
        if self.total_allocations == 0:
            return 0.0
        return self.successful_allocations / self.total_allocations
    
    @property
    def average_processing_time(self) -> float:
        """Calculate average processing time per allocation"""
        if self.successful_allocations == 0:
            return 0.0
        return self.total_processing_time / self.successful_allocations


@dataclass
class AttentionItem:
    """
    Individual item in attention focus with enhanced tracking and validation
    
    Attributes:
        id: Unique identifier for the attention item
        content: Stimulus content (validated and sanitized)
        salience: How attention-grabbing (0.0 to 1.0)
        activation: Current activation level (0.0 to 1.0)
        created_at: When the item was created
        last_updated: When the item was last modified
        priority: Task priority (0.0 to 1.0)
        effort_required: Cognitive effort needed (0.0 to 1.0)
        duration_seconds: How long it's been in focus
        correlation_id: For request tracing and debugging
    """
    id: str
    content: Any
    salience: float
    activation: float
    created_at: datetime
    last_updated: datetime
    priority: float = 0.5
    effort_required: float = 0.5
    duration_seconds: float = 0.0
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def __post_init__(self):
        """Validate attention item parameters after initialization"""
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate all parameters are within acceptable ranges"""
        validators = [
            (self.salience, 0.0, 1.0, "salience"),
            (self.activation, 0.0, 1.0, "activation"),
            (self.priority, 0.0, 1.0, "priority"),
            (self.effort_required, 0.0, 1.0, "effort_required"),
            (self.duration_seconds, 0.0, float('inf'), "duration_seconds")
        ]
        
        for value, min_val, max_val, name in validators:
            if not isinstance(value, (int, float)):
                raise InvalidStimulus(f"{name} must be numeric, got {type(value)}")
            if not (min_val <= value <= max_val):
                raise InvalidStimulus(f"{name} must be between {min_val} and {max_val}, got {value}")
        
        if not self.id or not isinstance(self.id, str):
            raise InvalidStimulus("id must be a non-empty string")
    
    def update_activation(self, delta: float) -> float:
        """
        Update activation level with validation and bounds checking
        
        Args:
            delta: Change in activation level
            
        Returns:
            New activation level
            
        Raises:
            InvalidStimulus: If delta is invalid
        """
        if not isinstance(delta, (int, float)):
            raise InvalidStimulus(f"Activation delta must be numeric, got {type(delta)}")
        
        old_activation = self.activation
        self.activation = max(0.0, min(1.0, self.activation + delta))
        self.last_updated = datetime.now()
        
        logger.debug(f"Updated activation for {self.id}: {old_activation:.3f} -> {self.activation:.3f} "
                    f"(delta: {delta:.3f})", extra={"correlation_id": self.correlation_id})
        
        return self.activation
    
    def age_seconds(self) -> float:
        """Get age in seconds since creation"""
        return (datetime.now() - self.created_at).total_seconds()
    
    def is_stale(self, max_age_seconds: float = 3600.0) -> bool:
        """Check if item is stale based on age"""
        return self.age_seconds() > max_age_seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "salience": self.salience,
            "activation": self.activation,
            "priority": self.priority,
            "effort_required": self.effort_required,
            "duration_seconds": self.duration_seconds,
            "age_seconds": self.age_seconds(),
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "correlation_id": self.correlation_id
        }

class AttentionMechanism:
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
    _shared_executor = ThreadPoolExecutor(max_workers=4)

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
            self._state_lock = threading.Lock()
            
            # Core attention state (thread-safe)
            self.focused_items: Dict[str, AttentionItem] = {}
            self.attention_history: deque = deque(maxlen=max_history_size)
            
            # Fatigue and resource management
            self.current_fatigue = 0.0
            self.total_cognitive_load = 0.0
            self.attention_capacity = 1.0
            
            # Performance tracking
            self.last_update = datetime.now()
            self.attention_episodes: List[Dict[str, Any]] = []
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
    
    def _auto_cleanup(self) -> None:
        """
        Automatically cleanup stale items and perform maintenance
        """
        current_time = datetime.now()
        
        # Check if cleanup is needed
        if (current_time - self._last_cleanup).total_seconds() < self.cleanup_interval:
            return
        
        items_removed = 0
        
        # Remove stale items
        stale_items = []
        for item_id, item in self.focused_items.items():
            if item.is_stale(max_age_seconds=3600.0):  # 1 hour default
                stale_items.append(item_id)
        
        for item_id in stale_items:
            self._remove_from_focus(item_id)
            items_removed += 1
        
        # Cleanup attention history - remove very old entries
        if len(self.attention_history) > 0:
            cutoff_time = current_time - timedelta(hours=24)  # Keep last 24 hours
            old_history = list(self.attention_history)
            recent_history = [
                event for event in old_history 
                if event.get('timestamp', current_time) > cutoff_time
            ]
            self.attention_history = deque(recent_history, maxlen=self.attention_history.maxlen)
        
        # Cleanup episodes - keep only recent ones
        if len(self.attention_episodes) > 500:
            self.attention_episodes = self.attention_episodes[-250:]
        
        self._last_cleanup = current_time
        
        if items_removed > 0:
            logger.debug(f"Auto-cleanup removed {items_removed} stale items")
    
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
        Returns (validated_params, effective_salience, dynamic_threshold)
        """
        validated_params = self._validate_allocation_params(
            stimulus_id, content, salience, novelty, priority, effort_required
        )
        with self._lock:
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
        """
        with self._lock:
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
        """
        with self._lock:
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
        """
        with self._lock:
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
    
    # ===== ASYNC SUPPORT METHODS =====
    
    async def allocate_attention_async(
        self,
        stimulus_id: str,
        content: Any,
        salience: float,
        novelty: float = 0.0,
        priority: float = 0.5,
        effort_required: float = 0.5
    ) -> Dict[str, Any]:
        """
        Async version of attention allocation for integration with cognitive agent.
        Uses a shared ThreadPoolExecutor to avoid race conditions and bottlenecks.
        Underlying sync methods are thread-safe via self._lock.
        """
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self._shared_executor,
            self.allocate_attention,
            stimulus_id, content, salience, novelty, priority, effort_required
        )
        return result
    
    async def update_attention_state_async(self, time_delta_seconds: Optional[float] = None) -> Dict[str, Any]:
        """Async version of attention state update. Uses shared executor for thread safety."""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self._shared_executor,
            self.update_attention_state,
            time_delta_seconds
        )
        return result
    
    async def rest_attention_async(self, duration_minutes: float = 1.0) -> Dict[str, float]:
        """Async version of attention rest. Uses shared executor for thread safety."""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self._shared_executor,
            self.rest_attention,
            duration_minutes
        )
        return result
    
    # ===== LIFECYCLE MANAGEMENT =====
    
    def shutdown(self) -> Dict[str, Any]:
        """
        Gracefully shutdown the attention mechanism
        
        Returns:
            Shutdown summary with final statistics
        """
        with self._lock:
            if self._is_shutdown:
                logger.warning("Attention mechanism already shutdown")
                return {"status": "already_shutdown"}
            self._is_shutdown = True
            # Collect final statistics
            final_stats = {
                "status": "shutdown_complete",
                "final_focused_items": len(self.focused_items),
                "total_episodes": len(self.attention_episodes),
                "final_cognitive_load": self.total_cognitive_load,
                "final_fatigue": self.current_fatigue,
                "shutdown_time": datetime.now().isoformat()
            }
            if self.metrics:
                final_stats.update({
                    "total_allocations": self.metrics.total_allocations,
                    "success_rate": self.metrics.success_rate,
                    "average_attention_score": self.metrics.average_attention_score,
                    "peak_cognitive_load": self.metrics.peak_cognitive_load,
                    "error_count": self.metrics.error_count
                })
            # Clear all focused items
            for item_id in list(self.focused_items.keys()):
                self._remove_from_focus(item_id)
            # Ensure all observer references are released for clean memory management
            self._observers.clear()
            logger.info(f"Attention mechanism shutdown complete. Final stats: {final_stats}")
            return final_stats
    
    # ===== OBSERVER PATTERN =====
    
    def add_observer(self, observer) -> None:
        """
        Add an observer for attention events
        
        Args:
            observer: Object with methods like on_attention_allocated, on_focus_lost, etc.
        """
        weak_ref = weakref.ref(observer)
        self._observers.append(weak_ref)
    
    def remove_observer(self, observer) -> bool:
        """Remove an observer"""
        for i, weak_ref in enumerate(self._observers):
            if weak_ref() is observer:
                del self._observers[i]
                return True
        return False
    
    def _notify_observers(self, event_type: str, data: Dict[str, Any]) -> None:
        """Notify all observers of an attention event"""
        # Clean up dead references
        self._observers = [ref for ref in self._observers if ref() is not None]
        for weak_ref in self._observers:
            observer = weak_ref()
            if observer is not None:
                try:
                    method_name = f"on_{event_type}"
                    if hasattr(observer, method_name):
                        getattr(observer, method_name)(data)
                except Exception as e:
                    logger.error(f"Error notifying observer {observer}: {e}")
        
        for weak_ref in self._observers:
            observer = weak_ref()
            if observer is not None:
                try:
                    method_name = f"on_{event_type}"
                    if hasattr(observer, method_name):
                        getattr(observer, method_name)(data)
                except Exception as e:
                    logger.error(f"Error notifying observer {observer}: {e}")
    
    # ===== ENHANCED SEARCH AND FILTERING =====
    
    def find_focused_items(
        self,
        min_salience: Optional[float] = None,
        min_priority: Optional[float] = None,
        max_age_seconds: Optional[float] = None,
        content_filter: Optional[str] = None
    ) -> List[AttentionItem]:
        """
        Find focused items matching specific criteria
        
        Args:
            min_salience: Minimum salience threshold
            min_priority: Minimum priority threshold  
            max_age_seconds: Maximum age in seconds
            content_filter: String to search for in content (case-insensitive)
            
        Returns:
            List of matching attention items
        """
        with self._lock:
            matches = []
            
            for item in self.focused_items.values():
                # Apply filters
                if min_salience is not None and item.salience < min_salience:
                    continue
                if min_priority is not None and item.priority < min_priority:
                    continue
                if max_age_seconds is not None and item.age_seconds() > max_age_seconds:
                    continue
                if content_filter is not None:
                    content_str = str(item.content).lower()
                    if content_filter.lower() not in content_str:
                        continue
                
                matches.append(item)
            
            return sorted(matches, key=lambda x: x.activation, reverse=True)
    
    def get_attention_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of attention state and history
        
        Returns:
            Detailed attention summary for monitoring and debugging
        """
        with self._lock:
            current_focus = self.get_attention_focus()
            
            # Calculate statistics
            if current_focus:
                avg_salience = sum(item['salience'] for item in current_focus) / len(current_focus)
                avg_priority = sum(item['priority'] for item in current_focus) / len(current_focus)
                total_effort = sum(item['effort_required'] for item in current_focus)
            else:
                avg_salience = avg_priority = total_effort = 0.0
            
            # Processing time statistics
            if self._processing_times:
                avg_processing_time = sum(self._processing_times) / len(self._processing_times)
                max_processing_time = max(self._processing_times)
            else:
                avg_processing_time = max_processing_time = 0.0
            
            summary = {
                "attention_state": {
                    "focused_items": len(self.focused_items),
                    "max_capacity": self.config.max_attention_items,
                    "capacity_utilization": len(self.focused_items) / self.config.max_attention_items,
                    "cognitive_load": self.total_cognitive_load,
                    "fatigue_level": self.current_fatigue,
                    "available_capacity": max(0.0, self.attention_capacity - self.total_cognitive_load)
                },
                "current_focus_stats": {
                    "average_salience": avg_salience,
                    "average_priority": avg_priority,
                    "total_effort_required": total_effort
                },
                "performance_metrics": {
                    "average_processing_time": avg_processing_time,
                    "max_processing_time": max_processing_time,
                    "total_episodes": len(self.attention_episodes),
                    "history_size": len(self.attention_history)
                },
                "configuration": {
                    "salience_threshold": self.config.salience_threshold,
                    "fatigue_decay_rate": self.config.fatigue_decay_rate,
                    "attention_recovery_rate": self.config.attention_recovery_rate,
                    "max_attention_items": self.config.max_attention_items
                },
                "error_counts": dict(self._error_counts),
                "last_update": self.last_update.isoformat(),
                "last_cleanup": self._last_cleanup.isoformat()
            }
            
            if self.metrics:
                summary["metrics"] = {
                    "total_allocations": self.metrics.total_allocations,
                    "successful_allocations": self.metrics.successful_allocations,
                    "success_rate": self.metrics.success_rate,
                    "capacity_exceeded_count": self.metrics.capacity_exceeded_count,
                    "focus_switches": self.metrics.focus_switches,
                    "average_attention_score": self.metrics.average_attention_score,
                    "peak_cognitive_load": self.metrics.peak_cognitive_load,
                    "error_count": self.metrics.error_count,
                    "average_processing_time": self.metrics.average_processing_time
                }
            
            return summary
    
    # ===== CONFIGURATION MANAGEMENT =====
    
    def update_config(self, new_config: AttentionConfig) -> bool:
        """
        Update attention configuration at runtime
        
        Args:
            new_config: New attention configuration
            
        Returns:
            True if update was successful
            
        Raises:
            AttentionError: If configuration is invalid
        """
        with self._lock:
            old_config = self.config  # Store old config before validation
            try:
                # Validate new configuration
                self.config = new_config
                self._validate_config()
                
                logger.info(f"Updated attention configuration: "
                           f"threshold {old_config.salience_threshold:.3f} -> {new_config.salience_threshold:.3f}, "
                           f"max_items {old_config.max_attention_items} -> {new_config.max_attention_items}")
                
                return True
                
            except Exception as e:
                # Restore old configuration on error
                self.config = old_config
                logger.error(f"Failed to update configuration: {e}")
                raise AttentionError(f"Configuration update failed: {e}") from e
    
    # ===== EXPORT/IMPORT METHODS =====
    
    def export_state(self) -> Dict[str, Any]:
        """
        Export current attention state for persistence or debugging
        
        Returns:
            Dictionary containing complete attention state
        """
        with self._lock:
            return {
                "version": "1.0",
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "max_attention_items": self.config.max_attention_items,
                    "salience_threshold": self.config.salience_threshold,
                    "fatigue_decay_rate": self.config.fatigue_decay_rate,
                    "attention_recovery_rate": self.config.attention_recovery_rate
                },
                "state": {
                    "current_fatigue": self.current_fatigue,
                    "total_cognitive_load": self.total_cognitive_load,
                    "attention_capacity": self.attention_capacity,
                    "last_update": self.last_update.isoformat()
                },
                "focused_items": [item.to_dict() for item in self.focused_items.values()],
                "metrics": {
                    "total_allocations": self.metrics.total_allocations if self.metrics else 0,
                    "successful_allocations": self.metrics.successful_allocations if self.metrics else 0,
                    "focus_switches": self.metrics.focus_switches if self.metrics else 0,
                    "average_attention_score": self.metrics.average_attention_score if self.metrics else 0.0,
                    "peak_cognitive_load": self.metrics.peak_cognitive_load if self.metrics else 0.0,
                    "error_count": self.metrics.error_count if self.metrics else 0
                },
                "recent_episodes": self.attention_episodes[-50:] if self.attention_episodes else [],
                "recent_history": list(self.attention_history)[-50:] if self.attention_history else []
            }
    
    # ===== CORE ATTENTION METHODS =====
    
    def update_attention_state(self, time_delta_seconds: Optional[float] = None) -> Dict[str, Any]:
        """
        Update attention state over time with thread safety and comprehensive error handling
        
        Args:
            time_delta_seconds: Time elapsed since last update
        
        Returns:
            Updated attention state information
            
        Raises:
            AttentionError: If state update fails
        """
        if self._is_shutdown:
            raise AttentionError("Attention mechanism is shutdown")
        
        with self._timed_operation("update_attention_state"):
            with self._lock:
                if time_delta_seconds is None:
                    time_delta_seconds = (datetime.now() - self.last_update).total_seconds()
                
                # Validate time delta
                if time_delta_seconds < 0:
                    logger.warning(f"Negative time delta: {time_delta_seconds}, using 0")
                    time_delta_seconds = 0
                elif time_delta_seconds > 3600:  # More than 1 hour
                    logger.warning(f"Large time delta: {time_delta_seconds}s, capping at 3600s")
                    time_delta_seconds = 3600
                
                # Decay attention activations over time
                items_to_remove = []
                for item_id, item in self.focused_items.items():
                    try:
                        # Attention naturally decays, with increased decay for older items

                        # Configurable decay curve modifiers
                        age_factor = min(1.0, item.age_seconds() / 600.0)
                        decay_rate = (
                            self.config.attention_decay_base +
                            self.current_fatigue * self.config.attention_decay_fatigue_scale +
                            age_factor * self.config.attention_decay_age_scale
                        )
                        decay = decay_rate * time_delta_seconds / 60.0  # Per minute

                        item.update_activation(-decay)
                        item.duration_seconds += time_delta_seconds

                        # Priority aging: slowly reduce priority for long-active items
                        priority_decay = time_delta_seconds / 3600.0  # lose 0.001 per second
                        item.priority = max(0.0, item.priority - priority_decay)

                        # Remove items that fall below activation threshold
                        if item.activation < 0.1:
                            items_to_remove.append(item_id)

                    except Exception as e:
                        logger.error(f"Error updating item {item_id}: {e}")
                        items_to_remove.append(item_id)
                
                # Remove items that lost attention
                for item_id in items_to_remove:
                    self._remove_from_focus(item_id)
                
                # Recover from fatigue during low-load periods
                if self.total_cognitive_load < 0.3:
                    recovery = self.config.attention_recovery_rate * time_delta_seconds / 60.0
                    self.current_fatigue = max(0.0, self.current_fatigue - recovery)
                
                # Update cognitive load based on current focus
                self._recalculate_cognitive_load()
                
                self.last_update = datetime.now()
                
                return self.get_attention_status()
    
    def get_attention_focus(self) -> List[Dict[str, Any]]:
        """
        Get current attention focus items in priority order
        
        Returns:
            List of attention items sorted by activation level
        """
        with self._lock:
            focus_items = []
            for item in sorted(self.focused_items.values(), 
                              key=lambda x: x.activation, reverse=True):
                focus_items.append({
                    "id": item.id,
                    "salience": item.salience,
                    "activation": item.activation,
                    "priority": item.priority,
                    "effort_required": item.effort_required,
                    "duration_seconds": item.duration_seconds,
                    "age_seconds": item.age_seconds(),
                    "correlation_id": item.correlation_id
                })
            return focus_items
    
    def rest_attention(self, duration_minutes: float = 1.0) -> Dict[str, float]:
        """
        Simulate attention rest/recovery period with validation and error handling
        
        Args:
            duration_minutes: Rest duration in minutes
        
        Returns:
            Recovery metrics and effectiveness data
            
        Raises:
            InvalidStimulus: If duration is invalid
            AttentionError: If rest operation fails
        """
        if self._is_shutdown:
            raise AttentionError("Attention mechanism is shutdown")
        
        # Validate duration
        if not isinstance(duration_minutes, (int, float)) or duration_minutes <= 0:
            raise InvalidStimulus(f"Duration must be positive number, got {duration_minutes}")
        
        if duration_minutes > 60:  # Safety limit
            logger.warning(f"Long rest duration {duration_minutes}min, capping at 60min")
            duration_minutes = 60.0
        
        with self._timed_operation("rest_attention"):
            with self._lock:
                initial_fatigue = self.current_fatigue
                initial_load = self.total_cognitive_load
                
                # Accelerated recovery during rest
                recovery_amount = self.config.attention_recovery_rate * duration_minutes * 2.0
                self.current_fatigue = max(0.0, self.current_fatigue - recovery_amount)
                
                # Reduce cognitive load as attention naturally disperses
                load_reduction = 0.1 * duration_minutes
                self.total_cognitive_load = max(0.0, self.total_cognitive_load - load_reduction)
                
                # Some items may lose focus during rest
                items_lost = 0
                for item_id, item in list(self.focused_items.items()):
                    if item.priority < 0.7:  # Low priority items fade during rest
                        decay = 0.2 * duration_minutes
                        try:
                            item.update_activation(-decay)
                            if item.activation < 0.2:
                                self._remove_from_focus(item_id)
                                items_lost += 1
                        except Exception as e:
                            logger.error(f"Error during rest for item {item_id}: {e}")
                            self._remove_from_focus(item_id)
                            items_lost += 1
                
                # Calculate recovery effectiveness
                fatigue_reduction = initial_fatigue - self.current_fatigue
                load_reduction_actual = initial_load - self.total_cognitive_load
                recovery_effective = fatigue_reduction > 0.01 or load_reduction_actual > 0.01
                
                logger.info(f"Attention rest: {duration_minutes}min, "
                           f"fatigue {initial_fatigue:.3f}->{self.current_fatigue:.3f}, "
                           f"load {initial_load:.3f}->{self.total_cognitive_load:.3f}, "
                           f"items lost: {items_lost}")
                
                return {
                    "duration_minutes": duration_minutes,
                    "fatigue_reduction": fatigue_reduction,
                    "load_reduction": load_reduction_actual,
                    "items_lost_focus": items_lost,
                    "recovery_effective": recovery_effective,
                    "final_fatigue": self.current_fatigue,
                    "final_load": self.total_cognitive_load,
                    "remaining_items": len(self.focused_items)
                }
    
    def get_attention_status(self) -> Dict[str, Any]:
        """
        Get comprehensive attention status with detailed metrics
        
        Returns:
            Dictionary containing current attention state and statistics
        """
        with self._lock:
            current_focus = self.get_attention_focus()
            return {
                "focused_items": len(self.focused_items),
                "max_capacity": self.config.max_attention_items,
                "capacity_utilization": len(self.focused_items) / self.config.max_attention_items,
                "cognitive_load": self.total_cognitive_load,
                "fatigue_level": self.current_fatigue,
                "attention_capacity": self.attention_capacity,
                "available_capacity": max(0.0, self.attention_capacity - self.total_cognitive_load),
                "focus_switches": self.metrics.focus_switches if self.metrics else 0,
                "salience_threshold": self.config.salience_threshold,
                "last_update": self.last_update.isoformat() if self.last_update else None,
                "current_focus": current_focus,
                "configuration": {
                    "max_attention_items": self.config.max_attention_items,
                    "salience_threshold": self.config.salience_threshold,
                    "fatigue_decay_rate": self.config.fatigue_decay_rate,
                    "attention_recovery_rate": self.config.attention_recovery_rate
                }
            }
