# Lean Test Replacement Plan

## Goal

Replace the current broad, slow, demo-heavy pytest suite with a small contract-focused suite that validates core functionality quickly and deterministically.

## Problems In The Current Suite

- Default discovery pulls in 59 test files and many heavyweight imports.
- Several files under `tests/` are demos or smoke scripts rather than assertion-driven tests.
- There are duplicate or superseded files such as `test_memory_integration.py` and `test_memory_integration_new.py`.
- Collection is slow because many tests import the full agent, neural, API, and executive stack.

## Replacement Strategy

### Phase 1: Create A New Default Suite

Use a new `tests/contracts/` directory as the default pytest target.

This suite should stay fast, deterministic, and focused on stable public contracts:

1. Intent classification contracts.
2. Context building contracts.
3. Chat service contracts.
4. Memory system contracts.

### Phase 2: Retire Legacy Discovery

Do not delete the old tests immediately.

Instead:

1. Change `pytest.ini` so default discovery points only at `tests/contracts`.
2. Treat the old suite as legacy/manual until each area is either replaced or intentionally discarded.

### Phase 3: Add Back Only High-Value Coverage

After the lean suite is stable, selectively reintroduce only tests that validate:

1. Public API compatibility.
2. One end-to-end smoke path for the full system.
3. Critical persistence behavior that cannot be covered with fakes.

## Planned Contract Tests

### 1. Intent Contracts

File: `tests/contracts/test_intent_contracts.py`

Coverage:

1. Goal creation messages classify as `goal_creation` with extracted entities.
2. Reminder messages classify as `reminder_request`.
3. Context-aware active goals are preserved on classifier instances.

### 2. Context Builder Contracts

File: `tests/contracts/test_context_builder_contracts.py`

Coverage:

1. Recent turn selection includes only user/system turns.
2. Reminder injection adds separate plan-step and reminder context items.
3. Fallback retrieval marks degraded mode when a memory backend fails.

### 3. Chat Service Contracts

File: `tests/contracts/test_chat_service_contracts.py`

Coverage:

1. `get_context_preview` returns stable payload shape.
2. Intent classifiers are session-scoped and reused per session.

### 4. Memory System Contracts

File: `tests/contracts/test_memory_system_contracts.py`

Coverage:

1. Important memories route to LTM.
2. Lower-importance memories route to STM.
3. `get_status` exposes system activity and operation metrics.

### 5. API Contracts

File: `tests/contracts/test_api_contracts.py`

Coverage:

1. Canonical `/agent/chat` returns the chat service payload.
2. Temporary consolidation threshold overrides are restored after the request.
3. Canonical `/agent/chat/preview` delegates to the chat service preview interface.

## Non-Goals For This Pass

- Rewriting all legacy scenario and integration tests.
- Full neural, ChromaDB, or end-to-end agent coverage in the default suite.
- Deleting old tests from the repository.

## Success Criteria

1. `pytest -q` runs only the new curated suite.
2. The default suite stays small and fast.
3. The new tests validate core behavior with clear assertions instead of demo-style logging.
4. Legacy tests remain available for manual migration if needed.

## Implementation Order

1. Add `tests/contracts/` and the new contract tests.
2. Update `pytest.ini` to point default discovery at `tests/contracts`.
3. Run `pytest -q` against the new suite.
4. Fix any failures and keep the suite green.

## Status

Completed:

1. Added the initial intent, context builder, chat service, and memory system contract tests.
2. Switched default pytest discovery to `tests/contracts`.
3. Refactored orchestration API import paths to be lazy so API modules can be imported quickly.
4. Added non-default smoke and persistence tiers.
5. Pruned the most obviously superseded legacy files, including zero-byte test files and older integration files replaced by `fixed` or `new` variants.
6. Removed the remaining zero-test scenario demo scripts that overlapped with the new smoke tier.

Next:

1. Continue pruning legacy files that are redundant but not yet zero-risk obvious.
2. Decide whether any remaining legacy suites should be archived outside `tests/` entirely.

## Additional Non-Default Tests

These do not run under the default `pytest -q` target.

### Smoke Tests

Directory: `tests/smoke/`

Purpose:

1. Exercise the real `CognitiveAgent.process_input` loop with lightweight fakes.
2. Catch integration regressions in the agent orchestration path.

### Persistence Tests

Directory: `tests/persistence/`

Purpose:

1. Validate persistence and reload behavior directly.
2. Keep storage-focused tests isolated from the fast contract suite.