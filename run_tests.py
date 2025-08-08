#!/usr/bin/env python3
"""
Test runner script that properly sets up the Python path for src-layout projects.
"""
import sys
from pathlib import Path

# Add src to the Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

if __name__ == "__main__":
    import pytest
    
    print("=" * 60)
    print("RUNNING UNIFIED MEMORY API TESTS")
    print("=" * 60)
    
    # Run the specific test file
    test_file = "tests/test_api_memory_unified.py"
    exit_code = pytest.main([test_file, "-v", "--tb=short"])
    
    print("=" * 60)
    if exit_code == 0:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED!")
    print("=" * 60)
    
    sys.exit(exit_code)
