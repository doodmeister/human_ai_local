# Test Organization

This directory contains both the active curated test suites and the older legacy suites that are being phased out of the default developer workflow.

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

## Legacy Suites

These remain in the repository for selective validation and migration work, but they are not part of the default `pytest -q` run:

- `integration/`
- `scenarios/`
- older top-level `test_*.py` files
- many historical `unit/` files

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

### Run Legacy Suites Intentionally
```bash
pytest tests/integration -q
pytest tests/scenarios -q
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
- Do not add new demo-style scripts as `test_*.py` files in legacy directories unless they are intended for pytest discovery.

## Contributing New Tests

When adding new tests, prefer the curated tiers first:

- stable public behavior → `contracts/`
- broader system path → `smoke/`
- persistence behavior → `persistence/`

Follow the naming convention: `test_[component_or_feature].py`
