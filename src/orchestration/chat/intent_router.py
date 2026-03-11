from __future__ import annotations

import time
from typing import Any, Callable, Dict, Iterable

from .metrics import metrics_registry


class ChatIntentRouter:
    """Plan and dispatch intent handlers for a chat turn."""

    def __init__(self, *, handler_priority: Iterable[str]) -> None:
        self._handler_priority = tuple(handler_priority)

    def plan_intent_execution(self, intent: Any) -> list[str]:
        allowed = set(self._handler_priority)
        ordered_candidates: list[str] = []
        if intent.intent_type in allowed:
            ordered_candidates.append(intent.intent_type)
        for intent_name, _conf in intent.secondary_intents:
            if intent_name in allowed and intent_name not in ordered_candidates:
                ordered_candidates.append(intent_name)
        return sorted(
            ordered_candidates,
            key=lambda name: (
                self._handler_priority.index(name),
                ordered_candidates.index(name),
            ),
        )

    @staticmethod
    def get_intent_confidence(intent: Any, intent_name: str) -> float:
        if intent_name == intent.intent_type:
            return intent.confidence
        for secondary_name, secondary_conf in intent.secondary_intents:
            if secondary_name == intent_name:
                return secondary_conf
        return 0.0

    def run_intent_handlers(
        self,
        *,
        intent: Any,
        message: str,
        session_id: str,
        resolve_handler: Callable[[str, Any, str, str], tuple[Callable[[], Any] | None, str | None]],
    ) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
        sections: list[Dict[str, Any]] = []
        execution_log: list[Dict[str, Any]] = []
        plan = self.plan_intent_execution(intent)
        for handler_name in plan:
            confidence = self.get_intent_confidence(intent, handler_name)
            handler_callable, unavailable_reason = resolve_handler(handler_name, intent, message, session_id)
            if handler_callable is None:
                execution_log.append(
                    {
                        "intent": handler_name,
                        "confidence": confidence,
                        "handled": False,
                        "response": None,
                        "error": unavailable_reason,
                        "duration_ms": 0.0,
                    }
                )
                continue

            metrics_registry.inc(f"intent_handler_{handler_name}_attempts_total")
            start = time.time()
            response_text = None
            error_text = None
            try:
                response_text = handler_callable()
                if response_text:
                    sections.append(
                        {
                            "intent": handler_name,
                            "confidence": confidence,
                            "content": response_text,
                        }
                    )
                    metrics_registry.inc(f"intent_handler_{handler_name}_handled_total")
            except Exception as exc:  # pragma: no cover
                error_text = str(exc)
                response_text = None
            duration_ms = (time.time() - start) * 1000.0
            execution_log.append(
                {
                    "intent": handler_name,
                    "confidence": confidence,
                    "handled": bool(response_text),
                    "response": response_text,
                    "error": error_text,
                    "duration_ms": duration_ms,
                }
            )
        return sections, execution_log