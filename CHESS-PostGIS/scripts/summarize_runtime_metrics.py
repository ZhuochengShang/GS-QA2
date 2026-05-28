#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def percentile(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def numeric(records, key):
    values = []
    for record in records:
        value = record.get(key)
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def summarize_values(records, key):
    values = numeric(records, key)
    if not values:
        return {
            f"{key}_mean": "",
            f"{key}_p95": "",
            f"{key}_max": "",
        }
    return {
        f"{key}_mean": round(mean(values), 3),
        f"{key}_p95": round(percentile(values, 0.95), 3),
        f"{key}_max": round(max(values), 3),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Summarize CHESS per-question runtime metrics."
    )
    parser.add_argument("result_dir", help="CHESS result directory containing -runtime_metrics.jsonl")
    parser.add_argument("--csv", dest="csv_path", help="Optional path to write per-question metrics CSV")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    metrics_path = result_dir / "-runtime_metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)

    records = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"No records found in {metrics_path}")

    summary = {
        "queries": len(records),
        "sql_success": sum(1 for r in records if r.get("final_sql_execution_status") == "success"),
        "sql_error": sum(1 for r in records if r.get("final_sql_execution_status") == "error"),
        "sql_not_run": sum(1 for r in records if r.get("final_sql_execution_status") == "not_run"),
        "mean_total_tokens": round(mean(numeric(records, "total_tokens")), 3) if numeric(records, "total_tokens") else "",
        "mean_prompt_tokens": round(mean(numeric(records, "prompt_tokens")), 3) if numeric(records, "prompt_tokens") else "",
        "mean_completion_tokens": round(mean(numeric(records, "completion_tokens")), 3) if numeric(records, "completion_tokens") else "",
        "mean_llm_calls": round(mean(numeric(records, "llm_calls")), 3) if numeric(records, "llm_calls") else "",
    }
    for key in ("pipeline_time_s", "generation_time_s", "llm_latency_s", "final_sql_execution_time_s"):
        summary.update(summarize_values(records, key))

    print(json.dumps(summary, indent=2))

    if args.csv_path:
        csv_path = Path(args.csv_path)
        fieldnames = sorted({key for record in records for key in record})
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        print(csv_path)


if __name__ == "__main__":
    main()
