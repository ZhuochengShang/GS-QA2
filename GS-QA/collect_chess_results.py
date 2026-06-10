#!/usr/bin/env python3
"""Compatibility wrapper for CHESS result collection.

Canonical implementation: ``baselines/chess/collect_chess_results.py``.
"""

from baselines.chess.collect_chess_results import *  # noqa: F401,F403
from baselines.chess.collect_chess_results import main


if __name__ == "__main__":
    main()

