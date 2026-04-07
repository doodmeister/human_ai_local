"""Lifecycle, async, observer, and time-update helpers for attention."""

from __future__ import annotations

import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager
from datetime import datetime, timedelta
from typing import Any, Dict, Optional
import weakref

from ...utils.logging import setup_logging
from .attention_state import AttentionStateMixin
from .exceptions import AttentionError, InvalidStimulus


logger = setup_logging(level="INFO", include_module=True)


class AttentionLifecycleMixin(AttentionStateMixin):
    """Lifecycle and maintenance behavior for the attention mechanism.

    Host classes must provide the mutable core operations implemented by
    ``AttentionMechanism`` such as allocation, removal, timed operations,
    and cognitive-load recalculation.
    """

    cleanup_interval: float
    _is_shutdown: bool
    _observers: list[weakref.ReferenceType]
    _shared_executor: Optional[ThreadPoolExecutor]
    _executor_lock: Any

    @classmethod
    def _get_shared_executor(cls) -> ThreadPoolExecutor:
        """Lazily create the shared executor used by async wrappers."""
        with cls._executor_lock:
            if cls._shared_executor is None:
                cls._shared_executor = ThreadPoolExecutor(max_workers=4)
            return cls._shared_executor

    @classmethod
    def shutdown_shared_executor(cls, wait: bool = True) -> bool:
        """Shutdown the process-wide shared executor and reset it for reuse."""
        with cls._executor_lock:
            executor = cls._shared_executor
            if executor is None:
                return False
            executor.shutdown(wait=wait)
            cls._shared_executor = None
            return True

    def allocate_attention(
        self,
        stimulus_id: str,
        content: Any,
        salience: float,
        novelty: float = 0.0,
        priority: float = 0.5,
        effort_required: float = 0.5,
    ) -> Dict[str, Any]:
        """Allocate attention for a stimulus.

        Concrete mechanisms provide the implementation.
        """
        raise NotImplementedError

    def _remove_from_focus(self, item_id: str) -> None:
        """Remove an item from focus.

        Concrete mechanisms provide the implementation.
        """
        raise NotImplementedError

    def _recalculate_cognitive_load(self) -> None:
        """Recalculate cognitive load from focused items.

        Concrete mechanisms provide the implementation.
        """
        raise NotImplementedError

    def _timed_operation(self, operation_name: str) -> AbstractContextManager[None]:
        """Track timing for an operation.

        Concrete mechanisms provide the implementation.
        """
        raise NotImplementedError

    def _auto_cleanup(self) -> None:
        """Automatically cleanup stale items and perform maintenance."""
        current_time = datetime.now()

        if (current_time - self._last_cleanup).total_seconds() < self.cleanup_interval:
            return

        items_removed = 0
        stale_items = []
        for item_id, item in self.focused_items.items():
            if item.is_stale(max_age_seconds=3600.0):
                stale_items.append(item_id)

        for item_id in stale_items:
            self._remove_from_focus(item_id)
            items_removed += 1

        if len(self.attention_history) > 0:
            cutoff_time = current_time - timedelta(hours=24)
            self.attention_history = deque(
                (
                    event
                    for event in self.attention_history
                    if self._coerce_event_timestamp(event.get("timestamp"), current_time) > cutoff_time
                ),
                maxlen=self.attention_history.maxlen,
            )

        self._last_cleanup = current_time

        if items_removed > 0:
            logger.debug(f"Auto-cleanup removed {items_removed} stale items")

    async def allocate_attention_async(
        self,
        stimulus_id: str,
        content: Any,
        salience: float,
        novelty: float = 0.0,
        priority: float = 0.5,
        effort_required: float = 0.5,
    ) -> Dict[str, Any]:
        """Async version of attention allocation."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._get_shared_executor(),
            self.allocate_attention,
            stimulus_id,
            content,
            salience,
            novelty,
            priority,
            effort_required,
        )

    async def update_attention_state_async(
        self, time_delta_seconds: Optional[float] = None
    ) -> Dict[str, Any]:
        """Async version of attention state update."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._get_shared_executor(),
            self.update_attention_state,
            time_delta_seconds,
        )

    async def rest_attention_async(self, duration_minutes: float = 1.0) -> Dict[str, float]:
        """Async version of attention rest."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._get_shared_executor(),
            self.rest_attention,
            duration_minutes,
        )

    def shutdown(self) -> Dict[str, Any]:
        """Gracefully shutdown the attention mechanism."""
        with self._lock:
            if self._is_shutdown:
                logger.warning("Attention mechanism already shutdown")
                return {"status": "already_shutdown"}
            self._is_shutdown = True
            final_stats = {
                "status": "shutdown_complete",
                "final_focused_items": len(self.focused_items),
                "total_episodes": len(self.attention_episodes),
                "final_cognitive_load": self.total_cognitive_load,
                "final_fatigue": self.current_fatigue,
                "shutdown_time": datetime.now().isoformat(),
            }
            if self.metrics:
                final_stats.update(
                    {
                        "total_allocations": self.metrics.total_allocations,
                        "success_rate": self.metrics.success_rate,
                        "average_attention_score": self.metrics.average_attention_score,
                        "peak_cognitive_load": self.metrics.peak_cognitive_load,
                        "error_count": self.metrics.error_count,
                    }
                )
            for item_id in list(self.focused_items.keys()):
                self._remove_from_focus(item_id)
            self._observers.clear()
            logger.info(f"Attention mechanism shutdown complete. Final stats: {final_stats}")
            return final_stats

    def add_observer(self, observer) -> None:
        """Add an observer for attention events."""
        weak_ref = weakref.ref(observer)
        self._observers.append(weak_ref)

    def remove_observer(self, observer) -> bool:
        """Remove an observer."""
        for index, weak_ref in enumerate(self._observers):
            if weak_ref() is observer:
                del self._observers[index]
                return True
        return False

    def _notify_observers(self, event_type: str, data: Dict[str, Any]) -> None:
        """Notify all observers of an attention event."""
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

    def update_attention_state(self, time_delta_seconds: Optional[float] = None) -> Dict[str, Any]:
        """Update attention state over time with thread safety and error handling."""
        if self._is_shutdown:
            raise AttentionError("Attention mechanism is shutdown")

        with self._timed_operation("update_attention_state"):
            with self._lock:
                if time_delta_seconds is None:
                    time_delta_seconds = (datetime.now() - self.last_update).total_seconds()

                if time_delta_seconds < 0:
                    logger.warning(f"Negative time delta: {time_delta_seconds}, using 0")
                    time_delta_seconds = 0
                elif time_delta_seconds > 3600:
                    logger.warning(f"Large time delta: {time_delta_seconds}s, capping at 3600s")
                    time_delta_seconds = 3600

                items_to_remove = []
                for item_id, item in self.focused_items.items():
                    try:
                        age_factor = min(1.0, item.age_seconds() / 600.0)
                        decay_rate = (
                            self.config.attention_decay_base
                            + self.current_fatigue * self.config.attention_decay_fatigue_scale
                            + age_factor * self.config.attention_decay_age_scale
                        )
                        decay = decay_rate * time_delta_seconds / 60.0

                        item.update_activation(-decay)
                        item.duration_seconds += time_delta_seconds

                        priority_decay = time_delta_seconds / 3600.0  # ~1.0 per hour (0.000278 per second)
                        item.priority = max(0.0, item.priority - priority_decay)

                        if item.activation < 0.1:
                            items_to_remove.append(item_id)

                    except Exception as e:
                        logger.error(f"Error updating item {item_id}: {e}")
                        items_to_remove.append(item_id)

                for item_id in items_to_remove:
                    self._remove_from_focus(item_id)

                if self.total_cognitive_load < 0.3:
                    recovery = self.config.attention_recovery_rate * time_delta_seconds / 60.0
                    self.current_fatigue = max(0.0, self.current_fatigue - recovery)

                self._recalculate_cognitive_load()
                self.last_update = datetime.now()
                return self.get_attention_status()

    def rest_attention(self, duration_minutes: float = 1.0) -> Dict[str, float]:
        """Simulate attention rest and recovery."""
        if self._is_shutdown:
            raise AttentionError("Attention mechanism is shutdown")

        if not isinstance(duration_minutes, (int, float)) or duration_minutes <= 0:
            raise InvalidStimulus(f"Duration must be positive number, got {duration_minutes}")

        if duration_minutes > 60:
            logger.warning(f"Long rest duration {duration_minutes}min, capping at 60min")
            duration_minutes = 60.0

        with self._timed_operation("rest_attention"):
            with self._lock:
                initial_fatigue = self.current_fatigue
                initial_load = self.total_cognitive_load

                recovery_amount = self.config.attention_recovery_rate * duration_minutes * 2.0
                self.current_fatigue = max(0.0, self.current_fatigue - recovery_amount)

                items_lost = 0
                for item_id, item in list(self.focused_items.items()):
                    if item.priority < 0.7:
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

                self._recalculate_cognitive_load()

                fatigue_reduction = initial_fatigue - self.current_fatigue
                load_reduction_actual = initial_load - self.total_cognitive_load
                recovery_effective = fatigue_reduction > 0.01 or load_reduction_actual > 0.01

                logger.info(
                    f"Attention rest: {duration_minutes}min, "
                    f"fatigue {initial_fatigue:.3f}->{self.current_fatigue:.3f}, "
                    f"load {initial_load:.3f}->{self.total_cognitive_load:.3f}, "
                    f"items lost: {items_lost}"
                )

                return {
                    "duration_minutes": duration_minutes,
                    "fatigue_reduction": fatigue_reduction,
                    "load_reduction": load_reduction_actual,
                    "items_lost_focus": items_lost,
                    "recovery_effective": recovery_effective,
                    "final_fatigue": self.current_fatigue,
                    "final_load": self.total_cognitive_load,
                    "remaining_items": len(self.focused_items),
                }