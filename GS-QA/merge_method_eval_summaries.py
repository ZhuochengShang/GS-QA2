#!/usr/bin/env python3
import argparse
import csv
import re
import sys
from pathlib import Path


csv.field_size_limit(sys.maxsize)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def truthy(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def as_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def task_sort_key(task: str):
    match = re.search(r"\d+", task or "")
    return int(match.group(0)) if match else 9999


def parse_baseline_path(path: Path, cache_root: Path):
    rel = path.relative_to(cache_root)
    model = rel.parts[0]
    task = rel.parts[1] if len(rel.parts) > 2 else ""
    name = path.name
    eval_kind = "parsed" if name.endswith("_parsed_eval.csv") else "text"
    prefix = name
    for suffix in ("_parsed_eval.csv", "_text_eval.csv"):
        if prefix.endswith(suffix):
            prefix = prefix[: -len(suffix)]
    baseline = prefix[len(model) + 1 :] if prefix.startswith(model + "_") else prefix
    method = f"{model}_{baseline}"
    return method, task, eval_kind


def tolerance_pass(row: dict, qtype: str, eval_kind: str, args) -> bool:
    if "passed" in row and row.get("passed") not in (None, ""):
        return truthy(row.get("passed"))
    if eval_kind == "text" or "name" in qtype or "multi_source" in qtype:
        f1 = as_float(row.get("F1"))
        return f1 is not None and f1 >= args.name_f1_tolerance
    if "loc" in qtype:
        distance_error = as_float(row.get("distance_error"))
        # baselines.py normalizes location error by 500 km and caps at 1.0.
        threshold = args.location_tolerance_m / 500000.0
        return distance_error is not None and distance_error <= threshold
    if "angle" in qtype:
        angle_error = as_float(row.get("angle_error"))
        # baselines.py normalizes angular error by 180 degrees.
        threshold = args.angle_tolerance_deg / 180.0
        return angle_error is not None and angle_error <= threshold
    if any(key in qtype for key in ("count", "area", "length", "distance")):
        relative_error = as_float(row.get("relative_error"))
        return relative_error is not None and relative_error <= args.numeric_relative_tolerance
    return False


def summarize_rows(rows: list[dict], method: str, task: str, eval_kind: str, args) -> list[dict]:
    groups = {}
    for row in rows:
        groups.setdefault(row.get("type", ""), []).append(row)
    output = []
    for qtype, items in groups.items():
        n = len(items)
        attempted = sum(1 for row in items if truthy(row.get("attempted")))
        summary = {
            "method": method,
            "task": task,
            "type": qtype,
            "eval_kind": eval_kind,
            "n": n,
            "attempted": attempted,
            "attempted_rate": attempted / n if n else 0,
        }
        for col in ("P", "R", "F1", "relative_error", "distance_error", "angle_error"):
            values = [as_float(row.get(col)) for row in items]
            avg = mean(values)
            if avg is not None:
                summary[f"mean_{col}"] = avg
        passed = sum(1 for row in items if tolerance_pass(row, qtype, eval_kind, args))
        summary["passed"] = passed
        summary["pass_rate"] = passed / n if n else 0
        output.append(summary)
    return output


def add_overall(rows: list[dict]) -> list[dict]:
    grouped = {}
    for row in rows:
        key = (row["method"], row["eval_kind"])
        grouped.setdefault(key, []).append(row)
    overall = []
    for (method, eval_kind), items in grouped.items():
        total_n = sum(int(row["n"]) for row in items)
        attempted = sum(int(row["attempted"]) for row in items)
        row = {
            "method": method,
            "task": "ALL",
            "type": "ALL",
            "eval_kind": eval_kind,
            "n": total_n,
            "attempted": attempted,
            "attempted_rate": attempted / total_n if total_n else 0,
        }
        for col in ("mean_P", "mean_R", "mean_F1", "mean_relative_error", "mean_distance_error", "mean_angle_error"):
            weighted = []
            for item in items:
                value = as_float(item.get(col))
                if value is not None:
                    weighted.extend([value] * int(item["n"]))
            avg = mean(weighted)
            if avg is not None:
                row[col] = avg
        if any("pass_rate" in item for item in items):
            passed = sum(int(item.get("passed", 0) or 0) for item in items)
            row["passed"] = passed
            row["pass_rate"] = passed / total_n if total_n else 0
        overall.append(row)
    return rows + overall


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "method", "task", "type", "eval_kind", "n", "attempted", "attempted_rate",
        "passed", "pass_rate", "mean_P", "mean_R", "mean_F1",
        "mean_relative_error", "mean_distance_error", "mean_angle_error",
    ]
    fieldnames = sorted({key for row in rows for key in row})
    fieldnames = [key for key in preferred if key in fieldnames] + [key for key in fieldnames if key not in preferred]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Merge CHESS and baseline evaluation CSVs into one summary table.")
    parser.add_argument("--cache-root", type=Path, default=Path("baselines/cache"))
    parser.add_argument("--chess-eval", type=Path, default=Path("CHESS/CHESS_IR_SS_CG_GEMINI/chess_sql_tolerance_eval.csv"))
    parser.add_argument("--output", type=Path, default=Path("baselines/combined_method_eval_summary.csv"))
    parser.add_argument("--numeric-relative-tolerance", type=float, default=0.05)
    parser.add_argument("--name-f1-tolerance", type=float, default=0.8)
    parser.add_argument("--location-tolerance-m", type=float, default=5.0)
    parser.add_argument("--angle-tolerance-deg", type=float, default=5.0)
    args = parser.parse_args()

    summaries = []
    for path in sorted(args.cache_root.glob("*/*/*_eval.csv")):
        method, task, eval_kind = parse_baseline_path(path, args.cache_root)
        summaries.extend(summarize_rows(read_csv(path), method, task, eval_kind, args))

    if args.chess_eval.exists():
        chess_rows = read_csv(args.chess_eval)
        tasks = sorted({row.get("task", "") for row in chess_rows}, key=task_sort_key)
        for task in tasks:
            rows = [row for row in chess_rows if row.get("task") == task]
            summaries.extend(summarize_rows(rows, "CHESS_IR_SS_CG_GEMINI", task, "tolerance", args))

    summaries = add_overall(summaries)
    summaries.sort(key=lambda row: (row["method"], task_sort_key(row["task"]), row["type"], row["eval_kind"]))
    write_csv(args.output, summaries)
    print(f"wrote {len(summaries)} rows -> {args.output}")


if __name__ == "__main__":
    main()
