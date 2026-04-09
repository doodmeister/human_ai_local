from __future__ import annotations

from collections import defaultdict
from typing import Any, DefaultDict, Mapping

from .interfaces import EventHandler


class InProcessEventBus:
    """Small in-process pub/sub bus for deterministic cognition phases."""

    def __init__(self) -> None:
        self._handlers: DefaultDict[str, list[EventHandler]] = defaultdict(list)
        self._failures: list[dict[str, Any]] = []

    def subscribe(self, event_name: str, handler: EventHandler) -> None:
        self._handlers[event_name].append(handler)

    def publish(self, event_name: str, payload: Mapping[str, Any]) -> None:
        for handler in tuple(self._handlers.get(event_name, ())):
            try:
                handler(payload)
            except Exception as exc:
                self._failures.append(
                    {
                        "event_name": event_name,
                        "error": str(exc),
                    }
                )

    def handler_count(self, event_name: str) -> int:
        return len(self._handlers.get(event_name, ()))

    def failures(self) -> list[dict[str, Any]]:
        return list(self._failures)
