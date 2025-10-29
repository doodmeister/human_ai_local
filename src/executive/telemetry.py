from __future__ import annotations

from typing import Any, Dict, Deque
from collections import deque

from .executive_core import EventBus

try:
    from src.chat.metrics import metrics_registry
except Exception:  # Lightweight fallback if chat metrics unavailable
    class _Dummy:
        def inc(self, *args, **kwargs):
            pass

        def observe_hist(self, *args, **kwargs):
            pass

        def state_set(self, *args, **kwargs):
            pass

        state: Dict[str, Any] = {}

    metrics_registry = _Dummy()  # type: ignore


RECENT_DECISIONS: Deque[Dict[str, Any]] = deque(maxlen=25)


def wire_eventbus_to_metrics(bus: EventBus) -> None:
    """Subscribe to Executive events and emit counters/histograms plus store recent decisions.

    Topics:
    - executive.turn_started -> mark event
    - executive.decision -> increment decisions & store rationale
    - executive.mode_changed -> increment mode changes
    - executive.turn_finished -> observe latency/tokens
    - executive.event_failure -> count failures
    """

    def _on_turn_started(e: Dict[str, Any]) -> None:
        try:
            metrics_registry.inc("executive_turns_started_total")
            if isinstance(e, dict):
                metrics_registry.state["executive_mode"] = e.get("mode")
        except Exception:
            pass

    def _on_decision(e: Dict[str, Any]) -> None:
        try:
            metrics_registry.inc("executive_decisions_total")
            if isinstance(e, dict):
                pol = e.get("policy")
                if pol:
                    metrics_registry.inc(f"executive_decisions_{pol}_total")
                RECENT_DECISIONS.append({
                    "policy": pol,
                    "option_id": e.get("option_id"),
                    "score": e.get("score"),
                    "rationale": e.get("rationale"),
                })
        except Exception:
            pass

    def _on_mode_changed(e: Dict[str, Any]) -> None:
        try:
            metrics_registry.inc("executive_mode_changes_total")
            if isinstance(e, dict):
                metrics_registry.state["executive_mode"] = e.get("to")
        except Exception:
            pass

    def _on_turn_finished(e: Dict[str, Any]) -> None:
        try:
            lat = float(e.get("latency_ms", 0.0)) if isinstance(e, dict) else 0.0
            metrics_registry.observe_hist("executive_turn_latency_ms", lat)
            metrics_registry.inc("executive_turns_total")
            if isinstance(e, dict) and "tokens" in e:
                metrics_registry.observe_hist("executive_tokens_used", float(e.get("tokens", 0)))
        except Exception:
            pass

    def _on_event_failure(e: Dict[str, Any]) -> None:
        try:
            metrics_registry.inc("executive_event_failures_total")
        except Exception:
            pass

    def _on_memory_utilization(e: Dict[str, Any]) -> None:
        try:
            if isinstance(e, dict):
                util = float(e.get("utilization", 0.0))
                metrics_registry.observe_hist("executive_memory_utilization", util)
        except Exception:
            pass

    bus.subscribe("executive.turn_started", _on_turn_started)
    bus.subscribe("executive.decision", _on_decision)
    bus.subscribe("executive.mode_changed", _on_mode_changed)
    bus.subscribe("executive.turn_finished", _on_turn_finished)
    bus.subscribe("executive.event_failure", _on_event_failure)
    bus.subscribe("executive.memory_utilization", _on_memory_utilization)
