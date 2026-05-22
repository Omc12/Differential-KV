# conftest.py — shared pytest fixtures available to all test files

import sys
from pathlib import Path
import pytest
import torch

# Ensure project root is importable in all tests
sys.path.insert(0, str(Path(__file__).resolve().parent))
