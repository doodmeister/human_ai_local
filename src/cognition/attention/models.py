"""Attention data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict
import uuid

from ...utils.logging import setup_logging
from .exceptions import InvalidStimulus


logger = setup_logging(level="INFO", include_module=True)


@dataclass
class AttentionMetrics:
    """Performance and usage metrics for attention mechanism."""

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
        """Calculate allocation success rate."""
        if self.total_allocations == 0:
            return 0.0
        return self.successful_allocations / self.total_allocations

    @property
    def average_processing_time(self) -> float:
        """Calculate average processing time per allocation."""
        if self.successful_allocations == 0:
            return 0.0
        return self.total_processing_time / self.successful_allocations


@dataclass
class AttentionItem:
    """Individual item in attention focus with enhanced tracking and validation."""

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

    def __post_init__(self) -> None:
        """Validate attention item parameters after initialization."""
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validate all parameters are within acceptable ranges."""
        validators = [
            (self.salience, 0.0, 1.0, "salience"),
            (self.activation, 0.0, 1.0, "activation"),
            (self.priority, 0.0, 1.0, "priority"),
            (self.effort_required, 0.0, 1.0, "effort_required"),
            (self.duration_seconds, 0.0, float("inf"), "duration_seconds"),
        ]

        for value, min_val, max_val, name in validators:
            if not isinstance(value, (int, float)):
                raise InvalidStimulus(f"{name} must be numeric, got {type(value)}")
            if not (min_val <= value <= max_val):
                raise InvalidStimulus(
                    f"{name} must be between {min_val} and {max_val}, got {value}"
                )

        if not self.id or not isinstance(self.id, str):
            raise InvalidStimulus("id must be a non-empty string")

    def update_activation(self, delta: float) -> float:
        """Update activation level with validation and bounds checking."""
        if not isinstance(delta, (int, float)):
            raise InvalidStimulus(f"Activation delta must be numeric, got {type(delta)}")

        old_activation = self.activation
        self.activation = max(0.0, min(1.0, self.activation + delta))
        self.last_updated = datetime.now()

        logger.debug(
            f"Updated activation for {self.id}: {old_activation:.3f} -> {self.activation:.3f} "
            f"(delta: {delta:.3f})",
            extra={"correlation_id": self.correlation_id},
        )

        return self.activation

    def age_seconds(self) -> float:
        """Get age in seconds since creation."""
        return (datetime.now() - self.created_at).total_seconds()

    def is_stale(self, max_age_seconds: float = 3600.0) -> bool:
        """Check if item is stale based on age."""
        return self.age_seconds() > max_age_seconds

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
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
            "correlation_id": self.correlation_id,
        }