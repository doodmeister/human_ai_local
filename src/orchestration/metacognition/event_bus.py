from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any, Callable, DefaultDict, Mapping

from .interfaces import EventHandler, Unsubscribe


class MetacognitiveEvent:
    CYCLE_STARTED = "metacognition.cycle.started"
    SNAPSHOT_CREATED = "metacognition.snapshot.created"
    CYCLE_COMPLETED = "metacognition.cycle.completed"


class InProcessEventBus:
    """Small synchronous in-process pub/sub bus for deterministic cognition phases.

    Handlers run synchronously on the publishing thread. Long-running handlers block
    the publish call and therefore the metacognitive cycle.
    """

    def __init__(
        self,
        *,
        max_failures: int = 500,
        on_failure: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._handlers: DefaultDict[str, list[EventHandler]] = defaultdict(list)
        self._failures: deque[dict[str, Any]] = deque(maxlen=max_failures)
        self._on_failure = on_failure

    def subscribe(self, event_name: str, handler: EventHandler) -> Unsubscribe:
        handlers = self._handlers[event_name]
        if handler not in handlers:
            handlers.append(handler)

        def _unsubscribe() -> None:
            self.unsubscribe(event_name, handler)

        return _unsubscribe

    def unsubscribe(self, event_name: str, handler: EventHandler) -> None:
        handlers = self._handlers.get(event_name)
        if not handlers:
            return
        try:
            handlers.remove(handler)
        except ValueError:
            return
        if not handlers:
            self._handlers.pop(event_name, None)

    def publish(self, event_name: str, payload: Mapping[str, Any]) -> int:
        failure_count = 0
        for handler in tuple(self._handlers.get(event_name, ())):
            handler_payload = dict(payload)
            try:
                handler(handler_payload)
            except Exception as exc:
                failure = {
                    "event_name": event_name,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "handler": getattr(handler, "__qualname__", repr(handler)),
                    "timestamp": time.time(),
                    "payload_keys": sorted(handler_payload.keys()),
                    "payload_preview": handler_payload,
                }
                self._failures.append(failure)
                failure_count += 1
                if self._on_failure is not None:
                    try:
                        self._on_failure(dict(failure))
                    except Exception:
                        pass
        return failure_count

    def handler_count(self, event_name: str) -> int:
        return len(self._handlers.get(event_name, ()))

    def total_handler_count(self) -> int:
        return sum(len(handlers) for handlers in self._handlers.values())

    def event_names(self) -> list[str]:
        return sorted(name for name, handlers in self._handlers.items() if handlers)

    def summary(self) -> dict[str, Any]:
        return {
            "event_count": len(self.event_names()),
            "total_handlers": self.total_handler_count(),
            "failure_count": len(self._failures),
            "events": {name: len(handlers) for name, handlers in self._handlers.items() if handlers},
        }

    def failures(self) -> list[dict[str, Any]]:
        return list(self._failures)

    def clear_failures(self) -> None:
        self._failures.clear()

    def clear(self) -> None:
        self._handlers.clear()
        self._failures.clear()
