"""
Production Resilience Module

Provides error recovery, circuit breakers, and retry mechanisms
for production hardening of the Human-AI Cognition Framework.
"""

import time
import logging
import functools
from typing import Callable, Any, Optional, TypeVar, Dict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5          # Failures before opening
    success_threshold: int = 2          # Successes to close from half-open
    timeout_seconds: float = 30.0       # Time before attempting recovery
    excluded_exceptions: tuple = ()      # Exceptions to not count as failures


@dataclass
class CircuitBreakerState:
    """State tracking for circuit breaker."""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_state_change: datetime = field(default_factory=datetime.now)


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.
    
    Prevents cascading failures by temporarily blocking calls to
    failing services.
    
    Usage:
        breaker = CircuitBreaker("memory_service")
        
        @breaker
        def call_memory_service():
            ...
    """
    
    _instances: Dict[str, "CircuitBreaker"] = {}
    _lock = Lock()
    
    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitBreakerState()
        self._lock = Lock()
    
    @classmethod
    def get_or_create(cls, name: str, config: Optional[CircuitBreakerConfig] = None) -> "CircuitBreaker":
        """Get existing circuit breaker or create new one."""
        with cls._lock:
            if name not in cls._instances:
                cls._instances[name] = cls(name, config)
            return cls._instances[name]
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state, considering timeout."""
        with self._lock:
            if self._state.state == CircuitState.OPEN:
                # Check if timeout has passed
                if self._state.last_failure_time:
                    elapsed = (datetime.now() - self._state.last_failure_time).total_seconds()
                    if elapsed >= self.config.timeout_seconds:
                        self._transition_to(CircuitState.HALF_OPEN)
            return self._state.state
    
    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._state.state
        self._state.state = new_state
        self._state.last_state_change = datetime.now()
        
        if new_state == CircuitState.CLOSED:
            self._state.failure_count = 0
            self._state.success_count = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._state.success_count = 0
        
        logger.info(f"Circuit '{self.name}' transitioned: {old_state.value} -> {new_state.value}")
    
    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            if self._state.state == CircuitState.HALF_OPEN:
                self._state.success_count += 1
                if self._state.success_count >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
    
    def record_failure(self, exception: Exception) -> None:
        """Record a failed call."""
        # Check if this exception type should be excluded
        if isinstance(exception, self.config.excluded_exceptions):
            return
        
        with self._lock:
            self._state.failure_count += 1
            self._state.last_failure_time = datetime.now()
            
            if self._state.state == CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.OPEN)
            elif self._state.state == CircuitState.CLOSED:
                if self._state.failure_count >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)
    
    def allow_request(self) -> bool:
        """Check if a request should be allowed."""
        return self.state != CircuitState.OPEN
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator to wrap function with circuit breaker."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            if not self.allow_request():
                raise CircuitOpenError(f"Circuit '{self.name}' is open")
            
            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure(e)
                raise
        
        return wrapper
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status for monitoring."""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.state.value,
                "failure_count": self._state.failure_count,
                "success_count": self._state.success_count,
                "last_failure": self._state.last_failure_time.isoformat() if self._state.last_failure_time else None,
                "last_state_change": self._state.last_state_change.isoformat(),
            }


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 10.0,
    exponential_base: float = 2.0,
    retryable_exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exponential_base: Base for exponential backoff
        retryable_exceptions: Tuple of exception types to retry
        on_retry: Optional callback called on each retry
    
    Usage:
        @retry_with_backoff(max_retries=3)
        def flaky_operation():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception: Optional[Exception] = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        delay = min(base_delay * (exponential_base ** attempt), max_delay)
                        
                        if on_retry:
                            on_retry(e, attempt + 1)
                        else:
                            logger.warning(
                                f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                                f"after {type(e).__name__}: {e}. Waiting {delay:.1f}s"
                            )
                        
                        time.sleep(delay)
            
            if last_exception is not None:
                raise last_exception
            raise RuntimeError("Unexpected retry loop exit without exception")
        
        return wrapper
    return decorator


def graceful_degradation(
    fallback_value: Any = None,
    fallback_func: Optional[Callable[..., T]] = None,
    log_errors: bool = True
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for graceful degradation on errors.
    
    Returns fallback value or calls fallback function on error.
    
    Args:
        fallback_value: Value to return on error
        fallback_func: Function to call on error (receives same args)
        log_errors: Whether to log errors
    
    Usage:
        @graceful_degradation(fallback_value=[])
        def get_recommendations():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.warning(
                        f"Graceful degradation for {func.__name__}: {type(e).__name__}: {e}"
                    )
                
                if fallback_func:
                    return fallback_func(*args, **kwargs)
                return fallback_value
        
        return wrapper
    return decorator


@dataclass
class HealthStatus:
    """Health status for a component."""
    name: str
    healthy: bool
    message: str = ""
    latency_ms: float = 0.0
    last_check: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)


class HealthChecker:
    """
    Health checker for monitoring system components.
    
    Usage:
        checker = HealthChecker()
        checker.register("database", check_database_health)
        checker.register("memory", check_memory_health)
        
        status = checker.check_all()
    """
    
    def __init__(self):
        self._checks: Dict[str, Callable[[], HealthStatus]] = {}
        self._last_results: Dict[str, HealthStatus] = {}
        self._lock = Lock()
    
    def register(self, name: str, check_func: Callable[[], HealthStatus]) -> None:
        """Register a health check function."""
        with self._lock:
            self._checks[name] = check_func
    
    def check(self, name: str) -> HealthStatus:
        """Run a specific health check."""
        if name not in self._checks:
            return HealthStatus(name=name, healthy=False, message="Unknown check")
        
        start = time.time()
        try:
            status = self._checks[name]()
            status.latency_ms = (time.time() - start) * 1000
            status.last_check = datetime.now()
        except Exception as e:
            status = HealthStatus(
                name=name,
                healthy=False,
                message=f"Check failed: {type(e).__name__}: {e}",
                latency_ms=(time.time() - start) * 1000
            )
        
        with self._lock:
            self._last_results[name] = status
        
        return status
    
    def check_all(self) -> Dict[str, HealthStatus]:
        """Run all health checks."""
        results = {}
        for name in self._checks:
            results[name] = self.check(name)
        return results
    
    def is_healthy(self) -> bool:
        """Check if all components are healthy."""
        results = self.check_all()
        return all(status.healthy for status in results.values())
    
    def get_summary(self) -> Dict[str, Any]:
        """Get health summary for API response."""
        results = self.check_all()
        
        return {
            "healthy": all(s.healthy for s in results.values()),
            "timestamp": datetime.now().isoformat(),
            "components": {
                name: {
                    "healthy": status.healthy,
                    "message": status.message,
                    "latency_ms": round(status.latency_ms, 2),
                    "details": status.details
                }
                for name, status in results.items()
            }
        }


# Global health checker instance
_health_checker = HealthChecker()


def get_health_checker() -> HealthChecker:
    """Get the global health checker instance."""
    return _health_checker


def register_health_check(name: str, check_func: Callable[[], HealthStatus]) -> None:
    """Register a health check with the global health checker."""
    _health_checker.register(name, check_func)


# Telemetry tracking
@dataclass
class TelemetryEvent:
    """A telemetry event for monitoring."""
    name: str
    timestamp: datetime
    duration_ms: float
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class TelemetryCollector:
    """
    Collects telemetry events for monitoring and analysis.
    
    Usage:
        telemetry = TelemetryCollector()
        
        with telemetry.track("operation_name"):
            do_something()
    """
    
    def __init__(self, max_events: int = 1000):
        self._events: list = []
        self._max_events = max_events
        self._lock = Lock()
        self._counters: Dict[str, int] = {}
    
    def record(self, event: TelemetryEvent) -> None:
        """Record a telemetry event."""
        with self._lock:
            self._events.append(event)
            if len(self._events) > self._max_events:
                self._events = self._events[-self._max_events:]
            
            # Update counters
            key = f"{event.name}_{'success' if event.success else 'failure'}"
            self._counters[key] = self._counters.get(key, 0) + 1
    
    def track(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        """Context manager for tracking an operation."""
        return _TelemetryContext(self, name, metadata or {})
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get telemetry metrics summary."""
        with self._lock:
            if not self._events:
                return {"events": 0, "counters": {}}
            
            # Calculate statistics per operation
            by_name: Dict[str, list] = {}
            for event in self._events:
                if event.name not in by_name:
                    by_name[event.name] = []
                by_name[event.name].append(event)
            
            stats = {}
            for name, events in by_name.items():
                durations = [e.duration_ms for e in events]
                successes = sum(1 for e in events if e.success)
                
                stats[name] = {
                    "count": len(events),
                    "success_rate": successes / len(events) if events else 0,
                    "avg_duration_ms": sum(durations) / len(durations) if durations else 0,
                    "max_duration_ms": max(durations) if durations else 0,
                    "min_duration_ms": min(durations) if durations else 0,
                }
            
            return {
                "events": len(self._events),
                "counters": dict(self._counters),
                "operations": stats
            }


class _TelemetryContext:
    """Context manager for telemetry tracking."""
    
    def __init__(self, collector: TelemetryCollector, name: str, metadata: Dict[str, Any]):
        self.collector = collector
        self.name = name
        self.metadata = metadata
        self.start_time: float = 0.0
        self.success = True
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.time() - self.start_time) * 1000
        self.success = exc_type is None
        
        if exc_type:
            self.metadata["error"] = f"{exc_type.__name__}: {exc_val}"
        
        self.collector.record(TelemetryEvent(
            name=self.name,
            timestamp=datetime.now(),
            duration_ms=duration_ms,
            success=self.success,
            metadata=self.metadata
        ))
        
        return False  # Don't suppress exceptions


# Global telemetry collector
_telemetry = TelemetryCollector()


def get_telemetry() -> TelemetryCollector:
    """Get the global telemetry collector."""
    return _telemetry
