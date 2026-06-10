#!/usr/bin/env python3
"""Compatibility wrapper for Text2SQL result collection.

Canonical implementation: ``baselines/text2sql/collect_text2sql_results.py``.
"""

from baselines.text2sql.collect_text2sql_results import *  # noqa: F401,F403
from baselines.text2sql.collect_text2sql_results import main


if __name__ == "__main__":
    main()

