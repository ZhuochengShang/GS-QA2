#!/usr/bin/env python3
import csv
import json
import pathlib
import re
from collections import Counter


def classify(err):
    e = (err or "").lower()
    if not e:
        return "Ran successfully"
    if "incorrect answer" in e:
        return "Ran successfully"
    if "task exceeded" in e or "timeout" in e or "timed out" in e or "statement timeout" in e:
        return "Timed out"
    if "syntax error" in e:
        return "Syntax error"
    if "function" in e and "does not exist" in e:
        return "Function does not exist"
    if "operator" in e and "does not exist" in e:
        return "Operator does not exist"
    if "column" in e and "does not exist" in e:
        return "Column does not exist"
    if "relation" in e and "does not exist" in e:
        return "Relation does not exist"
    if "missing from-clause" in e or "missing from clause" in e:
        return "Missing FROM clause"
    if "subquery" in e or "sub-query" in e or "more than one row returned" in e:
        return "Sub-query error"
    return "Other"


def text2sql_counts():
    counts = Counter()
    root = pathlib.Path("baselines/cache/gemini")
    for path in sorted(root.glob("T*/sql_exec.json"), key=lambda p: int(p.parent.name[1:])):
        for record in json.load(path.open()):
            errors = [item.get("error", "") for item in record.get("records", []) if item.get("error")]
            if not record.get("records"):
                counts["Other"] += 1
            elif any(classify(error) == "Timed out" for error in errors):
                counts["Timed out"] += 1
            elif errors:
                counts[classify(errors[0])] += 1
            else:
                counts["Ran successfully"] += 1
    return counts


def chess_counts():
    counts = Counter()
    root = pathlib.Path("/home/zshan011/CHESS/CHESS/results/dev/CHESS_IR_SS_CG_GEMINI")

    def latest_run_dir(result_dir):
        return sorted([p for p in result_dir.iterdir() if p.is_dir()])[-1]

    def final_error(history):
        for step in reversed(history):
            if not isinstance(step, dict):
                continue
            final_sql = step.get("final_SQL") or step.get("final_sql")
            if isinstance(final_sql, dict):
                return str(final_sql.get("exec_err", "") or "")
            if step.get("tool_name") == "execution_accuracy":
                for value in step.values():
                    if isinstance(value, dict) and "exec_err" in value:
                        return str(value.get("exec_err", "") or "")
        return ""

    for result_dir in sorted(root.glob("gsqa_T*_postgis"), key=lambda p: int(re.search(r"T(\d+)", p.name).group(1))):
        run_dir = latest_run_dir(result_dir)
        for history_path in run_dir.glob("*_*.json"):
            if not history_path.name.split("_", 1)[0].isdigit():
                continue
            counts[classify(final_error(json.load(history_path.open())))] += 1
    return counts


def main():
    methods = {
        "Text2SQL": text2sql_counts(),
        "CHESS-PostGIS": chess_counts(),
    }
    subcategories = [
        "Ran successfully",
        "Timed out",
        "Syntax error",
        "Function does not exist",
        "Operator does not exist",
        "Column does not exist",
        "Relation does not exist",
        "Missing FROM clause",
        "Sub-query error",
        "Other",
    ]
    output = pathlib.Path("baselines/text2sql_chess_error_summary.csv")
    with output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "subcategory", "Text2SQL", "CHESS-PostGIS"])
        for subcategory in ("Ran successfully", "Timed out"):
            writer.writerow(["Valid SQL", subcategory, methods["Text2SQL"][subcategory], methods["CHESS-PostGIS"][subcategory]])
        writer.writerow([
            "Valid SQL",
            "Total",
            methods["Text2SQL"]["Ran successfully"] + methods["Text2SQL"]["Timed out"],
            methods["CHESS-PostGIS"]["Ran successfully"] + methods["CHESS-PostGIS"]["Timed out"],
        ])
        for subcategory in subcategories[2:]:
            writer.writerow(["Invalid SQL", subcategory, methods["Text2SQL"][subcategory], methods["CHESS-PostGIS"][subcategory]])
        writer.writerow([
            "Invalid SQL",
            "Total",
            sum(methods["Text2SQL"][subcategory] for subcategory in subcategories[2:]),
            sum(methods["CHESS-PostGIS"][subcategory] for subcategory in subcategories[2:]),
        ])
    print(f"wrote {output}")
    print(output.read_text())


if __name__ == "__main__":
    main()
