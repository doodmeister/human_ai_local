# Test Organization

This directory contains the maintained default pytest suite for the repository.

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

## Current Layout

The default pytest target in `pytest.ini` is `tests/`, so `python -m pytest -q` discovers the maintained suite directly from this directory.

Current layout in practice:

- top-level `test_*.py` files for maintained feature and regression coverage
- focused subdirectories such as `executive/` and `unit/` for targeted runs
- a small number of sparse or historical subdirectories that are not the primary verification path

## Focused Specialized Tests

These remain under `tests/` because they are still assertion-driven and useful for targeted validation, and they are part of the default discovery unless explicitly excluded:

- `test_enhanced_ltm_comprehensive.py`
- `test_episodic_memory_integration.py`

## Archived Manual Legacy Tests

Manual-only legacy suites and older exploratory test files were moved out of `tests/` to:

- `archived_tests/manual_legacy/`

## Running Tests

### Default Maintained Suite
```bash
python -m pytest -q
```

### Run Targeted Areas
```bash
python -m pytest tests/executive -q
python -m pytest tests/unit -q
python -m pytest tests/test_chat_factory_integration.py -q
python -m pytest tests/test_memory_system_init_prospective.py -q
```

### Run Focused Specialized Files
```bash
python -m pytest tests/test_enhanced_ltm_comprehensive.py -q
python -m pytest tests/test_episodic_memory_integration.py -q
```

### Run Archived Legacy Material Intentionally
```bash
python -m pytest archived_tests/manual_legacy/integration -q
python -m pytest archived_tests/manual_legacy/scenarios -q
python -m pytest archived_tests/manual_legacy/unit -q
```

### Run A Specific File
```bash
python -m pytest tests/test_executive_api.py -q
python -m pytest tests/test_chat_factory_integration.py -q
python -m pytest tests/test_memory_system_init_prospective.py -q
```

## Placement Guidance

- New maintained tests should usually live at the top level of `tests/` unless there is a clear existing subdirectory for that area.
- Focused area tests can go in established subdirectories such as `tests/executive/` or `tests/unit/`.
- When a new directory convention is introduced, update this file and `pytest.ini` together.
- Do not add new demo-style scripts as `test_*.py` files. Historical manual-only material belongs under `archived_tests/manual_legacy/`, not in `tests/`.

## Contributing New Tests

When adding new tests, prefer the current maintained layout:

- stable public behavior and general regressions → top-level `tests/`
- area-specific coverage where a directory already exists → the relevant subdirectory under `tests/`
- manual or legacy-only validation → `archived_tests/manual_legacy/`

Follow the naming convention: `test_[component_or_feature].py`
