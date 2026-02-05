"""Back-compat shim for chat metrics.

Metrics live in `src.memory.metrics` to keep lower-layer visibility.
"""

from __future__ import annotations

from src.memory.metrics import (  # noqa: F401
    ChatMetricsRegistry,
    MetricsRegistry,
    metrics_registry,
    timed,
)
