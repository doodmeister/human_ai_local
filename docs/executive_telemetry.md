# Executive Telemetry & Metrics

The `ExecutiveController` emits lifecycle events via `EventBus` for auditability and metrics:

Topics:
- `executive.turn_started` {mode, goal, input}
- `executive.decision` {policy, option_id, score, rationale}
- `executive.mode_changed` {from, to, signal, fatigue, load}
- `executive.turn_finished` {latency_ms, tokens, mode, new_memories}

To bridge these events into existing Prometheus / metrics dashboards, call:

```python
from src.executive.telemetry import wire_eventbus_to_metrics
bus = EventBus()
wire_eventbus_to_metrics(bus)
controller = ExecutiveController(..., event_bus=bus)
```

Metrics registered (counters/histograms):
- `executive_turns_started_total`
- `executive_decisions_total` (+ per policy suffix, e.g. `executive_decisions_weighted_v1_total`)
- `executive_mode_changes_total`
- `executive_turn_latency_ms` (histogram)
- `executive_tokens_used` (histogram)
- `executive_turns_total`

Current mode is mirrored in `metrics_registry.state['executive_mode']` for UIs.

Extending telemetry:
1. Subscribe to `executive.decision` to record reward signals and feed bandit training.
2. Add error event types (`executive.error`) if you introduce recoverable failures.
3. Export selected metrics via FastAPI endpoint reusing existing `/agent/chat/performance` serialization.

Safety: Telemetry handlers swallow exceptions to avoid impacting the cognitive turn.
