#!/usr/bin/env python3
"""Compatibility wrapper for raster Text2SQL.

Canonical implementation: ``baselines/text2sql/run_raster_text2sql.py``.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.text2sql.run_raster_text2sql import *  # noqa: F401,F403
from baselines.text2sql.run_raster_text2sql import main


if __name__ == "__main__":
    main()
