# Metacognitive Enhancements

## Snapshot Interval
- Config key: `metacog_turn_interval` (default 5). Every N user turns a snapshot is generated and attached to chat response under `metacog`.

## Snapshot Contents
Fields may include:
- `ts`: timestamp
- `turn_counter`: cumulative user turn count for service instance
- `performance.latency_p95_ms` & `performance.degraded`
- `recent_consolidation_selectivity`: ltm_promotions / stm_store (if counters available)
- `promotion_age_p95_seconds` (if histogram recorded)
- `stm_utilization`, `stm_capacity` (best-effort)
- `last_user_turn_status` (stored | skipped)

## Context Injection
When performance degraded or STM utilization >= 85%, metacognitive advisory context items are appended (`source_system=metacog`).
- `reason=performance_degraded`
- `reason=stm_high_utilization`

### Metrics Counters
The following counters are emitted:
- `metacog_snapshots_total` – Number of snapshots generated.
- `metacog_advisory_items_total` – Total advisory context items injected.
- `metacog_stm_high_util_events_total` – Times STM high utilization advisory triggered.
- `metacog_performance_degraded_events_total` – Times performance degraded advisory triggered.

## Adaptive Consolidation Thresholds
If STM utilization high (>=0.85) or performance degraded, salience consolidation threshold is temporarily increased (+0.05 each condition, capped at 0.85) before consolidation decision, then restored.

## Adaptive Retrieval Limits
When performance is degraded or STM utilization high (>=0.85), `max_context_items` is temporarily reduced (multiplicative 0.75 each condition, floor 4) to lower retrieval & scoring overhead. Original value restored after the turn. Counter: `adaptive_retrieval_applied_total`.

## Dynamic Snapshot Interval
Metacog snapshot interval (`metacog_turn_interval`) tightens (-1 down to 2) when performance degraded or STM utilization high, and relaxes (+1 up to 10) when stable (not degraded and STM util < 0.70). Current interval exposed via performance endpoint under `metacog.interval`.

## Snapshot Persistence
If LTM available (must expose `add_item(content, metadata)`), each snapshot is stored with metadata type `meta_reflection`.

## Endpoint
`GET /agent/chat/metacog/status` returns the last snapshot:
```json
{"available": true, "snapshot": { ... }}
```

## Testing
New tests:
- `test_chat_metacog_snapshot.py` validates interval and context injection for degraded performance.
- `test_chat_metacog_metrics.py` validates counters for snapshots and advisory injections.
- `test_chat_adaptive_retrieval.py` validates adaptive retrieval limit application.
- `test_chat_dynamic_metacog_interval.py` validates dynamic interval tightening/relaxing.

## Implemented Enhancements
- Snapshot history ring buffer (configurable via `metacog_snapshot_history_size`).
- Structured Pydantic (v2-compatible) schema for snapshot validation.
- Adaptive retrieval limits & dynamic snapshot interval modulation.

## Future Improvements
- Additional adaptive levers (e.g., dynamic LTM promotion thresholds, selective stage skipping under severe degradation).
- Expose historical interval changes in a dedicated diagnostics endpoint.
- Persist snapshot ring buffer to disk between restarts (optional durability).
