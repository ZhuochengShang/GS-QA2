import json
import re
from pathlib import Path


for root, filename in (
    ("gemini", "sql_json_parse.json"),
    ("gemini_rag", "rag_json_parse.json"),
):
    print("---", root)
    for task_number in range(23, 29):
        path = Path("baselines/cache") / root / f"T{task_number}" / filename
        rows = json.loads(path.read_text())
        decimal_rows = []
        for row in rows:
            content = row.get("content", "") if isinstance(row, dict) else str(row)
            if re.search(r'"(?:count|area|length|distance)"\s*:\s*-?\d+\.\d+', content):
                decimal_rows.append(content[:180].replace("\n", " "))
        print(
            f"T{task_number}",
            "rows",
            len(rows),
            "decimal",
            len(decimal_rows),
            "sample",
            decimal_rows[:1],
        )
