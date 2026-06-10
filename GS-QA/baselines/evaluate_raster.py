#!/usr/bin/env python3
"""Compatibility wrapper for the raster evaluator.

Canonical implementation: ``baselines/evaluators/evaluate_raster.py``.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.evaluators.evaluate_raster import *  # noqa: F401,F403
from baselines.evaluators.evaluate_raster import main


if __name__ == "__main__":
    main()
