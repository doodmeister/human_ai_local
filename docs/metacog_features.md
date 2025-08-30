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

## Adaptive Consolidation Thresholds
If STM utilization high (>=0.85) or performance degraded, salience consolidation threshold is temporarily increased (+0.05 each condition, capped at 0.85) before consolidation decision, then restored.

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

## Future Improvements
- Persist multiple snapshot history with ring buffer.
- More granular adaptive strategies (adjust LTM promotion thresholds, retrieval limits).
- Use structured schema / pydantic model for snapshot.
