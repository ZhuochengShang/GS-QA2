#!/usr/bin/env python3
import argparse
import csv
import importlib
import json
import os
import sys
from pathlib import Path

from geopy.geocoders import Nominatim
from pyproj import Geod

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from baselines import baselines as runner

evaluate_mod = importlib.import_module("baselines.evaluate")

INVALID_NAME_REFS = {
    "T11": {12, 14},
    "T12": {8, 13, 15, 18, 19, 22, 24, 26, 34, 35, 36, 37, 45, 59, 63, 68, 73, 77, 89, 94, 95, 98},
}


def read_json(path):
    return json.loads(path.read_text())


def raw_records(cache_root, model, task, method):
    artifact_model = f"{model}_rag" if method == "rag" else model
    base = cache_root / artifact_model / task
    if method == "text2sql":
        return read_json(base / "sql_answer.json"), read_json(base / "sql_json_parse.json")
    return read_json(base / "rag_answer.json"), read_json(base / "rag_json_parse.json")


def questions_for_task(benchmark_root, task):
    questions = runner.load_benchmark_questions(benchmark_root, [task])
    paths = sorted(
        (benchmark_root / task).glob("*/question.json"),
        key=lambda path: int(path.parent.name),
    )
    for question, path in zip(questions, paths):
        question["_benchmark_index"] = int(path.parent.name)
    return questions


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_task(cache_root, benchmark_root, model, method, task, geocoder, geod):
    questions = questions_for_task(benchmark_root, task)
    answers, parsed_records = raw_records(cache_root, model, task, method)
    parsed_answers = [
        runner.extract_json_blocks(record.get("content", ""), record.get("id"))
        for record in parsed_records
    ]
    text_rows, parsed_rows = runner.evaluate_answers(
        questions, answers, parsed_answers, evaluate_mod, geocoder, geod, prefix=method
    )
    for question, text_row, parsed_row in zip(questions, text_rows, parsed_rows):
        for row in (text_row, parsed_row):
            row["task"] = task
            row["id"] = question["id"]
            row["benchmark_index"] = question["_benchmark_index"]
            row["type"] = question["type"]
    return text_rows, parsed_rows


def mean(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    return sum(values) / len(values) if values else None


def strict_accuracy(rows):
    return sum(bool(row.get("acc", 0)) for row in rows) / len(rows) if rows else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=Path("baselines/cache"))
    parser.add_argument("--benchmark-root", type=Path, default=Path("benchmark"))
    parser.add_argument("--model", default="gemini")
    parser.add_argument("--methods", nargs="+", default=["text2sql", "rag"])
    parser.add_argument("--tasks", nargs="+", default=[f"T{i}" for i in range(1, 29)])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    os.environ.setdefault("GSQA_USE_ADDRESS_CACHE", "0")
    geocoder = Nominatim(user_agent="SpatialQA_baseline_no_cache")
    geod = Geod(ellps="WGS84")

    for method in args.methods:
        all_text, all_parsed = [], []
        for task in args.tasks:
            text_rows, parsed_rows = evaluate_task(
                args.cache_root, args.benchmark_root, args.model, method, task, geocoder, geod
            )
            all_text.extend(text_rows)
            all_parsed.extend(parsed_rows)
            valid_text = text_rows
            valid_parsed = parsed_rows
            if task in INVALID_NAME_REFS:
                invalid = INVALID_NAME_REFS[task]
                valid_text = [row for row in text_rows if int(row["benchmark_index"]) not in invalid]
                valid_parsed = [row for row in parsed_rows if int(row["benchmark_index"]) not in invalid]
            attempted_text = [row for row in valid_text if row.get("attempted")]
            attempted_parsed = [row for row in valid_parsed if row.get("attempted")]
            print(
                method, task,
                "n", len(valid_parsed),
                "text_recall", mean(attempted_text, "R"),
                "parsed_f1", mean(attempted_parsed, "F1"),
                "strict_acc", strict_accuracy(valid_parsed),
                "attempted", len(attempted_parsed),
            )
        write_csv(args.output_dir / f"{method}_text_eval.csv", all_text)
        write_csv(args.output_dir / f"{method}_parsed_eval.csv", all_parsed)


if __name__ == "__main__":
    main()
