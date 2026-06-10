#!/usr/bin/env python3
"""Compatibility wrapper for raster RAG.

Canonical implementation: ``baselines/rag/run_raster_rag.py``.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.rag.run_raster_rag import *  # noqa: F401,F403
from baselines.rag.run_raster_rag import main


if __name__ == "__main__":
    main()
