"""Explicit opt-in import path for archived TensorFlow validation only."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "reference"))
