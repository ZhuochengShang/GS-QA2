#!/usr/bin/env python3
"""Compatibility wrapper for CHESS/PostGIS evaluation.

Canonical implementation: ``baselines/chess/evaluate_chess_sql.py``.
"""

from baselines.chess.evaluate_chess_sql import *  # noqa: F401,F403
from baselines.chess.evaluate_chess_sql import main


if __name__ == "__main__":
    main()

