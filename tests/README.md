# Test Organization

This directory contains all tests for the Human-AI Cognition Framework, organized into logical categories for easy navigation and maintenance.

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

## Directory Structure

### 📦 `unit/` - Unit Tests
Tests for individual components in isolation:
- `test_core.py` - Core system unit tests
- `test_memory_components.py` - Individual memory component tests
- `test_memory.py` - Memory system unit tests
- `test_memory_only.py` - Memory-only functionality tests
- `test_stm_isolated.py` - Short-term memory isolation tests
- `test_stm_fix.py` - STM specific fixes and tests
- `test_dummy_classes.py` - Mock/dummy class testing
- `test_import.py`, `test_imports.py` - Import validation tests
- `test_pytorch_compatibility.py` - PyTorch compatibility tests

### 🔗 `integration/` - Integration Tests
Tests for cross-component interactions and system integration:
- `test_basic_integration.py` - Basic cognitive agent integration
- `test_cognitive_integration.py` - Cognitive system integration
- `test_attention_integration.py` - Attention mechanism integration
- `test_memory_integration.py`, `test_memory_integration_new.py` - Memory system integration
- `test_sensory_integration.py` - Sensory processing integration
- `test_dpad_integration_fixed.py`, `test_dapd_integration.py` - DPAD neural network integration
- `test_lshn_integration.py` - LSHN neural network integration
- `test_vector_stm_integration.py` - Vector STM integration
- `test_vector_ltm.py` - Vector LTM integration
- `test_performance_optimization.py` - Performance optimization integration

### 🧠 `cognitive/` - Cognitive Behavior Tests
Tests for high-level cognitive behaviors and complete system demonstrations:
- `test_attention_demo.py` - Attention mechanism demonstration
- `test_cognitive_working.py` - Working cognitive system tests
- `test_final_integration_demo.py` - Complete system demonstration
- `test_enhanced_sensory_cognitive.py` - Enhanced cognitive processing
- `test_dream_consolidation_pipeline.py` - Dream consolidation testing
- `test_lshn_neural_replay.py` - Neural replay cognitive testing
- `test_complete_pipeline.py` - Complete cognitive pipeline
- `test_quick_pipeline.py` - Quick cognitive pipeline test

### 🐛 `debug/` - Debug & Development Tests
Tests for debugging, development, and troubleshooting:
- `test_debug.py` - Main debug utilities
- `test_async_debug.py`, `test_sync_debug.py` - Async/sync debugging
- `test_attention_debug.py` - Attention mechanism debugging
- `test_bypass.py` - Bypass testing for development
- `test_simple.py` - Simple functionality tests
- `test_status_sync.py` - Status synchronization tests
- `test_sensory_processing.py` - Sensory processing debugging
- `test_complete_pipeline_bak.py` - Backup pipeline tests

## Running Tests

### Run All Tests
```bash
# From project root
pytest tests/ -v
```

### Run Tests by Category
```bash
# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# Cognitive behavior tests only
pytest tests/cognitive/ -v

# Debug tests only
pytest tests/debug/ -v
```

### Run Specific Test Files
```bash
# Run a specific integration test
pytest tests/integration/test_basic_integration.py -v

# Run a specific cognitive test
pytest tests/cognitive/test_final_integration_demo.py -v
```

## Test Categories Explained

- **Unit Tests**: Test individual components in isolation to ensure they work correctly on their own
- **Integration Tests**: Test how different components work together and validate cross-component functionality
- **Cognitive Tests**: Test complete cognitive behaviors and demonstrate the system's human-like cognitive capabilities
- **Debug Tests**: Development and troubleshooting tests that help with debugging and system validation

## Contributing New Tests

When adding new tests, place them in the appropriate directory:
- Individual component tests → `unit/`
- Cross-component interaction tests → `integration/`
- High-level cognitive behavior tests → `cognitive/`
- Development/debugging utilities → `debug/`

Follow the naming convention: `test_[component_or_feature].py`
