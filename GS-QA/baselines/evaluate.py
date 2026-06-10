#!/usr/bin/env python3
"""Compatibility wrapper for the vector evaluator.

Canonical implementation: ``baselines/evaluators/evaluate_vector.py``.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.evaluators.evaluate_vector import *  # noqa: F401,F403


if __name__ == "__main__":
    raise SystemExit(
        "baselines/evaluate.py provides vector evaluation helper functions. "
        "Run baselines/baselines.py to generate vector parsed/text eval CSVs."
    )
