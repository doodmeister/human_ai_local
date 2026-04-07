"""State, reporting, export, and config helpers for attention."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from ...core.config import AttentionConfig
from ...utils.logging import setup_logging
from .exceptions import AttentionError
from .models import AttentionItem


logger = setup_logging(level="INFO", include_module=True)


class AttentionStateMixin:
    """Query, reporting, export, and config behavior for the attention mechanism.

    Host classes are expected to provide the core state initialized by
    ``AttentionMechanism.__init__``: locks, config, focused item storage,
    history/episode collections, metrics, and timestamps.
    """

    _lock: Any
    config: AttentionConfig
    focused_items: Dict[str, AttentionItem]
    attention_history: Any
    attention_episodes: Any
    metrics: Any
    current_fatigue: float
    total_cognitive_load: float
    attention_capacity: float
    _processing_times: Any
    _error_counts: Dict[str, int]
    last_update: datetime
    _last_cleanup: datetime

    @staticmethod
    def _coerce_event_timestamp(value: Any, fallback: datetime) -> datetime:
        """Normalize stored event timestamps for cleanup comparisons."""
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                return fallback
        return fallback

    def find_focused_items(
        self,
        min_salience: Optional[float] = None,
        min_priority: Optional[float] = None,
        max_age_seconds: Optional[float] = None,
        content_filter: Optional[str] = None,
    ) -> List[AttentionItem]:
        """Find focused items matching specific criteria."""
        with self._lock:
            matches = []

            for item in self.focused_items.values():
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
        """Get a comprehensive summary of attention state and history."""
        with self._lock:
            current_focus = self._get_attention_focus_unlocked()

            if current_focus:
                avg_salience = sum(item["salience"] for item in current_focus) / len(current_focus)
                avg_priority = sum(item["priority"] for item in current_focus) / len(current_focus)
                total_effort = sum(item["effort_required"] for item in current_focus)
            else:
                avg_salience = avg_priority = total_effort = 0.0

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
                    "available_capacity": max(0.0, self.attention_capacity - self.total_cognitive_load),
                },
                "current_focus_stats": {
                    "average_salience": avg_salience,
                    "average_priority": avg_priority,
                    "total_effort_required": total_effort,
                },
                "performance_metrics": {
                    "average_processing_time": avg_processing_time,
                    "max_processing_time": max_processing_time,
                    "total_episodes": len(self.attention_episodes),
                    "history_size": len(self.attention_history),
                },
                "configuration": {
                    "salience_threshold": self.config.salience_threshold,
                    "fatigue_decay_rate": self.config.fatigue_decay_rate,
                    "attention_recovery_rate": self.config.attention_recovery_rate,
                    "max_attention_items": self.config.max_attention_items,
                },
                "error_counts": dict(self._error_counts),
                "last_update": self.last_update.isoformat(),
                "last_cleanup": self._last_cleanup.isoformat(),
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
                    "average_processing_time": self.metrics.average_processing_time,
                }

            return summary

    def update_config(self, new_config: AttentionConfig) -> bool:
        """Update attention configuration at runtime."""
        with self._lock:
            old_config = self.config
            try:
                self.config = new_config
                self._validate_config()

                logger.info(
                    f"Updated attention configuration: "
                    f"threshold {old_config.salience_threshold:.3f} -> {new_config.salience_threshold:.3f}, "
                    f"max_items {old_config.max_attention_items} -> {new_config.max_attention_items}"
                )

                return True

            except Exception as e:
                self.config = old_config
                logger.error(f"Failed to update configuration: {e}")
                raise AttentionError(f"Configuration update failed: {e}") from e

    def _validate_config(self) -> None:
        """Validate the active attention configuration.

        Concrete mechanisms provide the implementation.
        """
        raise NotImplementedError

    def export_state(self) -> Dict[str, Any]:
        """Export current attention state for persistence or debugging."""
        with self._lock:
            return {
                "version": "1.0",
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "max_attention_items": self.config.max_attention_items,
                    "salience_threshold": self.config.salience_threshold,
                    "fatigue_decay_rate": self.config.fatigue_decay_rate,
                    "attention_recovery_rate": self.config.attention_recovery_rate,
                },
                "state": {
                    "current_fatigue": self.current_fatigue,
                    "total_cognitive_load": self.total_cognitive_load,
                    "attention_capacity": self.attention_capacity,
                    "last_update": self.last_update.isoformat(),
                },
                "focused_items": [item.to_dict() for item in self.focused_items.values()],
                "metrics": {
                    "total_allocations": self.metrics.total_allocations if self.metrics else 0,
                    "successful_allocations": self.metrics.successful_allocations if self.metrics else 0,
                    "focus_switches": self.metrics.focus_switches if self.metrics else 0,
                    "average_attention_score": self.metrics.average_attention_score if self.metrics else 0.0,
                    "peak_cognitive_load": self.metrics.peak_cognitive_load if self.metrics else 0.0,
                    "error_count": self.metrics.error_count if self.metrics else 0,
                },
                "recent_episodes": list(self.attention_episodes)[-50:] if self.attention_episodes else [],
                "recent_history": list(self.attention_history)[-50:] if self.attention_history else [],
            }

    def _get_attention_focus_unlocked(self) -> List[Dict[str, Any]]:
        """Return current attention focus items while the caller holds the lock."""
        focus_items = []
        for item in sorted(self.focused_items.values(), key=lambda x: x.activation, reverse=True):
            focus_items.append(
                {
                    "id": item.id,
                    "salience": item.salience,
                    "activation": item.activation,
                    "priority": item.priority,
                    "effort_required": item.effort_required,
                    "duration_seconds": item.duration_seconds,
                    "age_seconds": item.age_seconds(),
                    "correlation_id": item.correlation_id,
                }
            )
        return focus_items

    def get_attention_focus(self) -> List[Dict[str, Any]]:
        """Get current attention focus items in priority order."""
        with self._lock:
            return self._get_attention_focus_unlocked()

    def get_attention_status(self) -> Dict[str, Any]:
        """Get comprehensive attention status with detailed metrics."""
        with self._lock:
            current_focus = self._get_attention_focus_unlocked()
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
                    "attention_recovery_rate": self.config.attention_recovery_rate,
                },
            }