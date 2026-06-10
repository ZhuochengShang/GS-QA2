#!/usr/bin/env python3
import argparse
import csv
import json
import re
import sys
import time
from decimal import Decimal
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional


def latest_run_dir(result_dir: Path) -> Path:
    if (result_dir / "-predictions.json").exists():
        return result_dir
    candidates = [p for p in result_dir.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directory found under {result_dir}")
    return sorted(candidates)[-1]


def load_metrics(run_dir: Path) -> dict:
    metrics = defaultdict(lambda: {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "llm_latency_seconds": 0.0,
        "llm_calls": 0,
    })
    metrics_path = run_dir / "-llm_metrics.jsonl"
    if not metrics_path.exists():
        return metrics

    with metrics_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            qid = int(record["question_id"])
            metrics[qid]["prompt_tokens"] += int(record.get("prompt_tokens") or 0)
            metrics[qid]["completion_tokens"] += int(record.get("completion_tokens") or 0)
            metrics[qid]["total_tokens"] += int(record.get("total_tokens") or 0)
            metrics[qid]["llm_latency_seconds"] += float(record.get("latency_ms") or 0) / 1000.0
            metrics[qid]["llm_calls"] += 1
    return metrics


def sql_from_prediction(value) -> str:
    if not isinstance(value, str):
        return ""
    return value.split("\t----- bird -----\t", 1)[0].strip()


def sql_from_history(history: list) -> str:
    for step in reversed(history):
        final_sql = step.get("final_SQL") or step.get("final_sql")
        if isinstance(final_sql, dict):
            sql = final_sql.get("PREDICTED_SQL") or final_sql.get("SQL")
            if sql:
                return str(sql).strip()
        sql = step.get("SQL")
        if sql:
            return str(sql).strip()
    return ""


def extract_sql_candidate(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"(?is)corrected query\W+(WITH\b.*?;|SELECT\b.*?;)", text)
    if match:
        return " ".join(match.group(1).split())
    match = re.search(r"(?is)```sql\s*(.*?)```", text)
    if match:
        return " ".join(match.group(1).split())
    match = re.search(r"(?s)\b(WITH\b.*?;|SELECT\b.*?;)", text)
    if match:
        return " ".join(match.group(1).split())
    return text


def final_status(history: list) -> tuple[str, str]:
    for step in reversed(history):
        final_sql = step.get("final_SQL") or step.get("final_sql")
        if isinstance(final_sql, dict):
            return str(final_sql.get("exec_res", "")), str(final_sql.get("exec_err", ""))
        if step.get("tool_name") == "execution_accuracy":
            for value in step.values():
                if isinstance(value, dict) and "exec_res" in value:
                    return str(value.get("exec_res", "")), str(value.get("exec_err", ""))
    return "", ""


def convert_value(value):
    if isinstance(value, Decimal):
        return float(value)
    return value


def records_from_cursor(cur, rows):
    columns = [desc.name for desc in cur.description] if cur.description else []
    records = []
    for row in rows:
        record = {}
        for column, value in zip(columns, row):
            record[column] = convert_value(value)
        records.append(record)
    return records


def execute_sql(conn, sql: str, statement_timeout_seconds: float = 360.0):
    if not conn or not sql:
        return ""
    with conn.cursor() as cur:
        timeout_ms = max(1, int(statement_timeout_seconds * 1000))
        cur.execute(f"SET LOCAL statement_timeout = {timeout_ms}")
        cur.execute(sql)
        try:
            rows = cur.fetchall()
        except Exception:
            rows = []
        return records_from_cursor(cur, rows)


def table_for_name_column(name_column: str) -> Optional[str]:
    return {
        "poi_name": "pois",
        "park_name": "parks",
        "lake_name": "lakes",
        "road_name": "roads",
        "region_name": "regions",
    }.get(name_column)


def enrich_records_by_name(conn, records: list):
    if not conn or not records:
        return records

    lookup_requests = {}
    record_lookups = []
    for record in records:
        if not isinstance(record, dict) or "geometry" in record:
            record_lookups.append(None)
            continue

        name_items = [
            (key, record.get(key))
            for key in ("poi_name", "park_name", "lake_name", "road_name", "region_name")
            if record.get(key)
        ]
        if not name_items and len(record) == 1:
            only_key, only_value = next(iter(record.items()))
            # Common aliases from generated SQL such as SELECT poi_name AS name.
            if only_key in ("name", "?column?") and only_value:
                name_items = [
                    ("poi_name", only_value),
                    ("park_name", only_value),
                    ("lake_name", only_value),
                    ("road_name", only_value),
                    ("region_name", only_value),
                ]

        record_lookups.append(name_items)
        for name_column, name_value in name_items:
            table = table_for_name_column(name_column)
            if table:
                lookup_requests.setdefault((table, name_column), set()).add(name_value)

    lookup_results = {}
    with conn.cursor() as cur:
        for (table, name_column), values in lookup_requests.items():
            value_list = list(values)
            if not value_list:
                continue
            placeholders = ", ".join(["%s"] * len(value_list))
            try:
                cur.execute(
                    f"SELECT *, ST_AsText(geometry::geometry) AS geometry FROM {table} WHERE {name_column} IN ({placeholders})",
                    value_list,
                )
                rows = cur.fetchall()
            except Exception:
                conn.rollback()
                continue
            for row in records_from_cursor(cur, rows):
                row_name = row.get(name_column)
                lookup_results.setdefault((name_column, row_name), row)

    enriched = []
    for record, name_items in zip(records, record_lookups):
        if not name_items:
            enriched.append(record)
            continue
        enriched_record = None
        for name_column, name_value in name_items:
            enriched_record = lookup_results.get((name_column, name_value))
            if enriched_record:
                break
        enriched.append(enriched_record or record)
    return enriched


def answer_text_from_records(records: list) -> str:
    values = []
    preferred = ["poi_name", "park_name", "lake_name", "road_name", "region_name", "name"]
    for record in records:
        if not isinstance(record, dict):
            continue
        value = None
        for key in preferred:
            if record.get(key) is not None:
                value = record[key]
                break
        if value is None and record:
            value = next(iter(record.values()))
        if value is not None:
            values.append(str(value))
    return "\n".join(values)


def write_baseline_answers(rows: list, benchmark_root: Path, baseline_key: str):
    for row in rows:
        qid = int(row["question_id"])
        qdir = benchmark_root / f"{qid:3d}"
        baseline_path = qdir / "baseline_answers.json"
        if baseline_path.exists():
            data = json.loads(baseline_path.read_text())
        else:
            qdir.mkdir(parents=True, exist_ok=True)
            data = {}

        try:
            parsed_answer = json.loads(row["answer"]) if row["answer"] else []
        except json.JSONDecodeError:
            parsed_answer = []

        data[baseline_key] = {
            "text": {
                "answer": row["answer_text"],
                "scores": {}
            },
            "parsed": {
                "answer": parsed_answer,
                "scores": {}
            },
            "sql": row["sql"],
            "metadata": {
                "pipeline_time_seconds": row["pipeline_time_seconds"],
                "llm_latency_seconds": row["llm_latency_seconds"],
                "prompt_tokens": row["prompt_tokens"],
                "completion_tokens": row["completion_tokens"],
                "total_tokens": row["total_tokens"],
                "llm_calls": row["llm_calls"],
                "final_exec_res": row["final_exec_res"],
                "final_exec_err": row["final_exec_err"],
            }
        }
        baseline_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def load_question_text(question_root, qid: int) -> str:
    if question_root is None:
        return ""
    question_path = question_root / f"{qid:3d}" / "question.json"
    if not question_path.exists():
        return ""
    return json.loads(question_path.read_text()).get("question", "")


def write_question_results(rows: list, question_root, output_root: Path):
    for row in rows:
        qid = int(row["question_id"])
        qdir = output_root / f"{qid:3d}"
        qdir.mkdir(parents=True, exist_ok=True)
        try:
            answers = json.loads(row["answer"]) if row["answer"] else []
        except json.JSONDecodeError:
            answers = []
        payload = {
            "question": load_question_text(question_root, qid),
            "sql": row["sql"],
            "answers": answers,
            "metadata": {
                "question_id": qid,
                "db_id": row["db_id"],
                "pipeline_time_seconds": row["pipeline_time_seconds"],
                "llm_latency_seconds": row["llm_latency_seconds"],
                "prompt_tokens": row["prompt_tokens"],
                "completion_tokens": row["completion_tokens"],
                "total_tokens": row["total_tokens"],
                "llm_calls": row["llm_calls"],
                "final_exec_res": row["final_exec_res"],
                "final_exec_err": row["final_exec_err"],
            }
        }
        (qdir / "question.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def connect_postgres(pg_params: dict):
    import psycopg
    return psycopg.connect(**pg_params)


def collect_history_row(path: Path, index: int, total: int, args, metrics, predictions, conn=None, pg_params=None):
    qid = int(path.name.split("_", 1)[0])
    db_id = path.stem.split("_", 1)[1]
    history = json.loads(path.read_text())
    pipeline_time = sum(float(step.get("execution_time") or 0) for step in history if isinstance(step, dict))
    metric = metrics[qid]
    sql_raw = sql_from_history(history) or sql_from_prediction(predictions.get(str(qid)))
    sql = extract_sql_candidate(sql_raw)
    exec_res, exec_err = final_status(history)

    answer = ""
    answer_text = ""
    local_conn = None
    active_conn = conn
    if args.execute and pipeline_time <= args.max_seconds:
        started_at = time.monotonic()
        if args.progress:
            print(f"[{index}/{total}] qid={qid} db={db_id} executing SQL...", flush=True)
        try:
            if active_conn is None:
                local_conn = connect_postgres(pg_params)
                active_conn = local_conn
            records = execute_sql(active_conn, sql, args.statement_timeout_seconds)
            if args.enrich_name_results:
                records = enrich_records_by_name(active_conn, records)
            answer = json.dumps(records, default=str, ensure_ascii=False)
            answer_text = answer_text_from_records(records)
            active_conn.commit()
            if args.progress:
                elapsed = time.monotonic() - started_at
                print(f"[{index}/{total}] qid={qid} ok rows={len(records)} elapsed={elapsed:.2f}s", flush=True)
        except Exception as exc:
            if active_conn is not None:
                active_conn.rollback()
            answer = f"EXECUTION_ERROR: {exc}"
            if args.progress:
                elapsed = time.monotonic() - started_at
                print(f"[{index}/{total}] qid={qid} error elapsed={elapsed:.2f}s: {exc}", file=sys.stderr, flush=True)
        finally:
            if local_conn is not None:
                local_conn.close()

    row = {
        "question_id": qid,
        "db_id": db_id,
        "pipeline_time_seconds": round(pipeline_time, 3),
        "llm_latency_seconds": round(metric["llm_latency_seconds"], 3),
        "prompt_tokens": metric["prompt_tokens"],
        "completion_tokens": metric["completion_tokens"],
        "total_tokens": metric["total_tokens"],
        "llm_calls": metric["llm_calls"],
        "final_exec_res": exec_res,
        "final_exec_err": exec_err,
        "sql": sql,
        "sql_raw": sql_raw,
        "answer": answer,
        "answer_text": answer_text,
    }
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--max-seconds", type=float, default=360.0)
    parser.add_argument("--execute", action="store_true", help="Execute SQL to fill answer; requires psycopg/PostGIS access.")
    parser.add_argument("--pg-host", default="localhost")
    parser.add_argument("--pg-database", default="gsqa")
    parser.add_argument("--pg-user", default="")
    parser.add_argument("--pg-password", default="")
    parser.add_argument("--pg-port", type=int, default=5432)
    parser.add_argument("--statement-timeout-seconds", type=float, default=360.0, help="Per-SQL PostgreSQL statement timeout while --execute is enabled.")
    parser.add_argument("--jobs", type=int, default=1, help="Number of question SQL executions to run concurrently. Use cautiously with PostGIS.")
    parser.add_argument("--progress", action="store_true", help="Print per-question execution progress and elapsed time.")
    parser.add_argument("--write-baseline-answers", action="store_true", help="Merge executed SQL answers into benchmark/T*/<id>/baseline_answers.json.")
    parser.add_argument("--benchmark-root", type=Path, default=None, help="Benchmark task directory, for example benchmark/T1.")
    parser.add_argument("--baseline-key", default="CHESS_IR_SS_CG_GEMINI", help="Top-level key to write in baseline_answers.json.")
    parser.add_argument("--write-question-results", action="store_true", help="Write question-style JSON files containing question, predicted SQL, and SQL answer rows.")
    parser.add_argument("--question-root", type=Path, default=None, help="Original question directory, for example benchmark/T1.")
    parser.add_argument("--question-output-root", type=Path, default=None, help="Output directory for question-style SQL result JSON files.")
    parser.add_argument("--enrich-name-results", action="store_true", help="For name-only SQL results, look up full OSM rows with geometry.")
    args = parser.parse_args()

    run_dir = latest_run_dir(args.result_dir)
    metrics = load_metrics(run_dir)
    predictions_path = run_dir / "-predictions.json"
    predictions = json.loads(predictions_path.read_text()) if predictions_path.exists() else {}

    pg_params = {
        "host": args.pg_host,
        "dbname": args.pg_database,
        "user": args.pg_user,
        "password": args.pg_password,
        "port": args.pg_port,
    }

    rows = []
    ignored_rows = []
    history_paths = [
        p for p in run_dir.glob("*_*.json")
        if p.name.split("_", 1)[0].isdigit()
    ]
    sorted_history_paths = sorted(history_paths, key=lambda p: int(p.name.split("_", 1)[0]))
    total_history_paths = len(sorted_history_paths)

    if args.jobs > 1:
        if args.execute and args.progress:
            print(f"Executing SQL with {args.jobs} worker connections", flush=True)
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = [
                executor.submit(collect_history_row, path, index, total_history_paths, args, metrics, predictions, None, pg_params)
                for index, path in enumerate(sorted_history_paths, start=1)
            ]
            for future in as_completed(futures):
                row = future.result()
                if row["pipeline_time_seconds"] <= args.max_seconds:
                    rows.append(row)
                else:
                    ignored_rows.append(row)
    else:
        conn = connect_postgres(pg_params) if args.execute else None
        try:
            for index, path in enumerate(sorted_history_paths, start=1):
                row = collect_history_row(path, index, total_history_paths, args, metrics, predictions, conn, pg_params)
                if row["pipeline_time_seconds"] <= args.max_seconds:
                    rows.append(row)
                else:
                    ignored_rows.append(row)
        finally:
            if conn is not None:
                conn.close()

    rows.sort(key=lambda row: int(row["question_id"]))
    ignored_rows.sort(key=lambda row: int(row["question_id"]))

    fieldnames = list(rows[0].keys() if rows else ignored_rows[0].keys())
    output_csv = run_dir / f"collected_results_under_{int(args.max_seconds)}s.csv"
    output_json = run_dir / f"collected_results_under_{int(args.max_seconds)}s.json"
    ignored_csv = run_dir / f"ignored_over_{int(args.max_seconds)}s.csv"

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    output_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False))

    with ignored_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ignored_rows)

    if args.write_baseline_answers:
        if not args.execute:
            raise ValueError("--write-baseline-answers requires --execute so SQL results are available.")
        if args.benchmark_root is None:
            raise ValueError("--write-baseline-answers requires --benchmark-root, for example benchmark/T1.")
        write_baseline_answers(rows, args.benchmark_root, args.baseline_key)
    if args.write_question_results:
        if not args.execute:
            raise ValueError("--write-question-results requires --execute so SQL results are available.")
        if args.question_output_root is None:
            raise ValueError("--write-question-results requires --question-output-root.")
        write_question_results(rows, args.question_root, args.question_output_root)

    print(f"run_dir: {run_dir}")
    print(f"kept: {len(rows)}")
    print(f"ignored_over_{int(args.max_seconds)}s: {len(ignored_rows)}")
    print(f"csv: {output_csv}")
    print(f"json: {output_json}")
    print(f"ignored_csv: {ignored_csv}")
    if args.write_baseline_answers:
        print(f"baseline_answers: updated {len(rows)} files under {args.benchmark_root}")
    if args.write_question_results:
        print(f"question_results: wrote {len(rows)} files under {args.question_output_root}")


if __name__ == "__main__":
    main()
