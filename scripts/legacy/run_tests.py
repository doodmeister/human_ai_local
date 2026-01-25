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
    raise SystemExit(
        "run_tests.py is not an entrypoint anymore. "
        "Use `python -m pytest -q` (or the VS Code task 'Test: pytest (-q)')."
    )
