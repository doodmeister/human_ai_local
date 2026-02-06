# Refactor & Cleanup Report (2026-02-06)

## Scope
Systematic scan for unused/legacy functionality, placeholders, and cleanup candidates. Recommendations below only suggest removals when functionality is clearly legacy or unused.

## Completed Cleanup (Feb 6, 2026)

### API & docs cleanup
- Removed legacy `/api/*` aliases across servers, tests, and docs.
- Updated docs to canonical routes and scrubbed archived `/api` references.

### Legacy scripts removed
- [scripts/legacy/geroge_test.py](scripts/legacy/geroge_test.py#L1-L210)
- [scripts/legacy/run_tests.py](scripts/legacy/run_tests.py#L1-L20)
- [scripts/legacy/analyze_code.py](scripts/legacy/analyze_code.py#L1-L120)
- [scripts/legacy/george_cli.py](scripts/legacy/george_cli.py#L1-L40)

### Test suite pruning
- Removed manual/utility scripts from tests: [tests/test_executive_api.py](tests/test_executive_api.py#L1-L220), [tests/test_api_simple.py](tests/test_api_simple.py#L1-L160), [tests/test_simple_api.py](tests/test_simple_api.py#L1-L120).
- Removed empty or duplicate stubs: [tests/test_vector_ltm.py](tests/test_vector_ltm.py), [tests/test_dpad_integration_fixed.py](tests/test_dpad_integration_fixed.py).
- Removed heavy integration tests:
  - [tests/integration/test_performance_optimization.py](tests/integration/test_performance_optimization.py#L1-L200)
  - [tests/integration/test_lshn_integration.py](tests/integration/test_lshn_integration.py#L1-L120)
  - [tests/integration/test_dpad_integration_fixed.py](tests/integration/test_dpad_integration_fixed.py#L1-L200)
  - [tests/integration/test_vector_only_memory.py](tests/integration/test_vector_only_memory.py#L1-L140)
  - [tests/integration/test_vector_stm_integration.py](tests/integration/test_vector_stm_integration.py#L1-L140)
  - [tests/integration/test_proactive_recall.py](tests/integration/test_proactive_recall.py#L1-L140)
  - [tests/integration/test_semantic_memory_integration.py](tests/integration/test_semantic_memory_integration.py#L1-L80)
- Removed duplicate/older integration tests: [tests/integration/test_dapd_integration.py](tests/integration/test_dapd_integration.py#L1-L120), [tests/integration/test_memory_integration.py](tests/integration/test_memory_integration.py#L1-L120).
- Removed script-style episodic integration test: [tests/test_episodic_integration_fixed.py](tests/test_episodic_integration_fixed.py#L1-L120).

## Remaining Items (Not Resolved)

1) Placeholder response path in `ChatService`
- Replace deterministic placeholder with production response strategy.
- See [src/orchestration/chat/chat_service.py](src/orchestration/chat/chat_service.py#L1921-L1934).

2) HTN goal manager adapter usage
- Decide whether HTN is runtime‑integrated or experimental.
- See [src/executive/goals/htn_goal_manager_adapter.py](src/executive/goals/htn_goal_manager_adapter.py#L1-L80).

3) Legacy test shims for old module paths
- Keep until legacy imports are fully removed; then delete.
- See [tests/conftest.py](tests/conftest.py#L1-L70).
