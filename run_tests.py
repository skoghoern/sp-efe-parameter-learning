#!/usr/bin/env python
"""Run the test suite for the two paper environments.

Usage: ``uv run python run_tests.py [extra pytest args]``
"""

import sys
from pathlib import Path

import pytest

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    sys.exit(pytest.main([str(root / "tests"), "-v", *sys.argv[1:]]))
