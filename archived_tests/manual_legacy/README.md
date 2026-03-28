# Manual Legacy Test Archive

This directory contains historical tests that were removed from the active `tests/` tree because they are manual-only, exploratory, slow, or tied to older API and integration paths.

These files are kept for reference, migration work, and one-off validation, but they are not part of the normal developer workflow.

## What Lives Here

- `integration/`: historical integration coverage for older vector, semantic, proactive recall, and neural paths.
- `scenarios/`: broader scenario-driven and demonstration-style tests.
- `unit/`: older unit-era tests retained for archaeology and selective migration.
- top-level archived `test_*.py` files: older API, procedural, STM, and LTM validation files that are no longer part of the supported active suite.

## Active Test Entry Points

Use these first:

```bash
python -m pytest -q
python -m pytest tests/executive -q
python -m pytest tests/unit -q
python -m pytest tests/test_chat_factory_integration.py -q
python -m pytest tests/test_enhanced_ltm_comprehensive.py -q
python -m pytest tests/test_episodic_memory_integration.py -q
```

## Running Archived Material Intentionally

```bash
python -m pytest archived_tests/manual_legacy/integration -q
python -m pytest archived_tests/manual_legacy/scenarios -q
python -m pytest archived_tests/manual_legacy/unit -q
```

If an archived test becomes important again, migrate it back into the maintained `tests/` suite rather than growing new legacy-only coverage under `tests/`.