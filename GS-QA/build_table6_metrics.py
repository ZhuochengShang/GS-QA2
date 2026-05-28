#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path


csv.field_size_limit(sys.maxsize)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


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
    if task == "ALL":
        return 999
    return int(task[1:]) if task.startswith("T") and task[1:].isdigit() else 998


def parse_baseline_path(path: Path, cache_root: Path):
    rel = path.relative_to(cache_root)
    model = rel.parts[0]
    task = rel.parts[1]
    name = path.name
    eval_kind = "parsed" if name.endswith("_parsed_eval.csv") else "text"
    prefix = name.replace("_parsed_eval.csv", "").replace("_text_eval.csv", "")
    baseline = prefix[len(model) + 1:] if prefix.startswith(model + "_") else prefix
    return f"{model}_{baseline}", task, eval_kind


def output_family(qtype: str):
    if "name" in qtype or "multi_source" in qtype:
        return "entity_name"
    if "loc" in qtype:
        return "location"
    if "angle" in qtype:
        return "direction"
    if "area" in qtype:
        return "area"
    if "length" in qtype:
        return "length"
    if "distance" in qtype:
        return "distance"
    if "count" in qtype:
        return "count"
    return "other"


def baseline_summaries(cache_root: Path):
    grouped = {}
    for path in sorted(cache_root.glob("*/*/*_eval.csv")):
        method, task, eval_kind = parse_baseline_path(path, cache_root)
        for row in read_csv(path):
            key = (method, task, row.get("type", ""), output_family(row.get("type", "")))
            item = grouped.setdefault(key, {"text": [], "parsed": []})
            item[eval_kind].append(row)

    rows = []
    for (method, task, qtype, family), parts in grouped.items():
        text_rows = parts["text"]
        parsed_rows = parts["parsed"]
        n = max(len(text_rows), len(parsed_rows))
        out = {
            "method": method,
            "task": task,
            "type": qtype,
            "output_family": family,
            "n": n,
        }
        if text_rows:
            out["text_recall"] = mean(as_float(row.get("R")) for row in text_rows)
            out["text_f1"] = mean(as_float(row.get("F1")) for row in text_rows)
        if parsed_rows:
            if family in ("entity_name", "location", "direction"):
                out["parsed_recall"] = mean(as_float(row.get("R")) for row in parsed_rows)
                out["parsed_f1"] = mean(as_float(row.get("F1")) for row in parsed_rows)
            if family == "location":
                out["distance_error"] = mean(as_float(row.get("distance_error")) for row in parsed_rows)
                out["distance_error_unit"] = "normalized_by_500km"
            if family == "direction":
                out["angle_error"] = mean(as_float(row.get("angle_error")) for row in parsed_rows)
                out["angle_error_unit"] = "normalized_by_180deg"
            if family in ("area", "length", "distance", "count"):
                out["relative_error"] = mean(as_float(row.get("relative_error")) for row in parsed_rows)
        rows.append(out)
    return rows


def chess_summaries(chess_eval: Path):
    grouped = {}
    for row in read_csv(chess_eval):
        qtype = row.get("type", "")
        key = ("CHESS_IR_SS_CG_GEMINI", row.get("task", ""), qtype, output_family(qtype))
        grouped.setdefault(key, []).append(row)

    rows = []
    for (method, task, qtype, family), items in grouped.items():
        out = {
            "method": method,
            "task": task,
            "type": qtype,
            "output_family": family,
            "n": len(items),
        }
        if family in ("entity_name", "location", "direction"):
            out["parsed_recall"] = mean(as_float(row.get("recall")) for row in items)
            out["parsed_f1"] = mean(as_float(row.get("score")) for row in items if row.get("metric") == "name_f1")
        if family == "location":
            out["distance_error"] = mean(as_float(row.get("score")) for row in items if row.get("metric") == "location_error_m")
            out["distance_error_unit"] = "meters"
        if family == "direction":
            out["angle_error"] = mean(as_float(row.get("score")) for row in items if row.get("metric") == "angle_error_deg")
            out["angle_error_unit"] = "degrees"
        if family in ("area", "length", "distance", "count"):
            out["relative_error"] = mean(as_float(row.get("score")) for row in items if "relative_error" in row.get("metric", ""))
        rows.append(out)
    return rows


def add_overall(rows: list[dict]):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["method"], row["output_family"]), []).append(row)
    output = list(rows)
    for (method, family), items in grouped.items():
        total = sum(int(row["n"]) for row in items)
        out = {"method": method, "task": "ALL", "type": "ALL", "output_family": family, "n": total}
        for col in ("text_recall", "text_f1", "parsed_recall", "parsed_f1", "distance_error", "angle_error", "relative_error"):
            values = []
            for row in items:
                value = as_float(row.get(col))
                if value is not None:
                    values.extend([value] * int(row["n"]))
            avg = mean(values)
            if avg is not None:
                out[col] = avg
        units = {row.get("distance_error_unit") for row in items if row.get("distance_error_unit")}
        if len(units) == 1:
            out["distance_error_unit"] = units.pop()
        units = {row.get("angle_error_unit") for row in items if row.get("angle_error_unit")}
        if len(units) == 1:
            out["angle_error_unit"] = units.pop()
        output.append(out)
    return output


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "method", "task", "type", "output_family", "n",
        "text_recall", "text_f1", "parsed_recall", "parsed_f1",
        "distance_error", "distance_error_unit", "angle_error", "angle_error_unit",
        "relative_error",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Build Table-6-style metrics for CHESS and baselines.")
    parser.add_argument("--cache-root", type=Path, default=Path("baselines/cache"))
    parser.add_argument("--chess-eval", type=Path, default=Path("CHESS/CHESS_IR_SS_CG_GEMINI/chess_sql_tolerance_eval_strict_geo.csv"))
    parser.add_argument("--output", type=Path, default=Path("baselines/table6_metrics_summary.csv"))
    args = parser.parse_args()

    rows = baseline_summaries(args.cache_root)
    if args.chess_eval.exists():
        rows.extend(chess_summaries(args.chess_eval))
    rows = add_overall(rows)
    rows.sort(key=lambda row: (row["method"], row["output_family"], task_sort_key(row["task"]), row["type"]))
    write_csv(args.output, rows)
    print(f"wrote {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
