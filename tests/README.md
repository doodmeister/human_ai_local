# Test Organization

This directory now contains only the active test suites and a small set of focused specialized memory tests.

## ⚠️ Important: Test File Naming Convention

**Actual test files** that pytest should run MUST follow these patterns:
- `test_*.py` - Files starting with "test_"
- `*_test.py` - Files ending with "_test"

**Utility/debug scripts** should NOT match these patterns to avoid confusion:
- ✅ Use: `debug_*.py`, `util_*.py`, `script_*.py`
- ❌ Avoid: `test_*.py` for non-test scripts

Debug and one-off scripts have been removed from this directory. If you need temporary
debugging scripts, create them in a `scratch/` subdirectory (gitignored) or use
descriptive names like `debug_chromadb.py` which are excluded by `.gitignore`.

## Active Test Tiers

### `contracts/` - Default Fast Suite
Contract tests for stable, high-value behavior.

This is the default pytest target configured in `pytest.ini`.

Covers:
- intent classification
- context building
- chat service behavior
- memory system routing
- canonical API contracts

### `smoke/` - Non-Default Orchestration Checks
Smoke tests exercise broader paths, such as the full agent processing loop, but stay out of the default fast suite.

### `persistence/` - Non-Default Storage Checks
Persistence tests verify reload and storage behavior directly, such as episodic JSON-backed persistence.

## Focused Specialized Tests

These remain under `tests/` because they are still assertion-driven and useful for targeted validation, but they are not part of the default `pytest -q` run:

- `test_enhanced_ltm_comprehensive.py`
- `test_episodic_memory_integration.py`

## Archived Manual Legacy Tests

Manual-only legacy suites and older exploratory test files were moved out of `tests/` to:

- `archived_tests/manual_legacy/`

## Running Tests

### Default Fast Suite
```bash
pytest -q
```

### Run By Tier
```bash
pytest tests/contracts -q
pytest tests/smoke -q
pytest tests/persistence -q
```

### Run Focused Specialized Files
```bash
pytest tests/test_enhanced_ltm_comprehensive.py -q
pytest tests/test_episodic_memory_integration.py -q
```

### Run Archived Legacy Material Intentionally
```bash
pytest archived_tests/manual_legacy/integration -q
pytest archived_tests/manual_legacy/scenarios -q
pytest archived_tests/manual_legacy/unit -q
```

### Run A Specific File
```bash
pytest tests/contracts/test_api_contracts.py -q
pytest tests/smoke/test_cognitive_agent_smoke.py -q
pytest tests/persistence/test_episodic_json_persistence.py -q
```

## Placement Guidance

- New default developer tests should usually go in `contracts/`.
- Broader system checks that are still valuable but slower should go in `smoke/`.
- Storage and reload verification should go in `persistence/`.
- Focused specialized tests that are still maintained can live at the top level of `tests/`.
- Do not add new demo-style scripts as `test_*.py` files. Historical manual-only material belongs under `archived_tests/manual_legacy/`, not in `tests/`.

## Contributing New Tests

When adding new tests, prefer the curated tiers first:

- stable public behavior → `contracts/`
- broader system path → `smoke/`
- persistence behavior → `persistence/`

Follow the naming convention: `test_[component_or_feature].py`
