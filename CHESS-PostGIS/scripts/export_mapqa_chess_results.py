#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import psycopg
from psycopg.rows import dict_row


def load_jsonl(path: Path) -> dict[int, dict]:
    rows = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = row.get("question_id")
            if qid is not None:
                rows[int(qid)] = row
    return rows


def clean_prediction_sql(value: str) -> str:
    if not value:
        return ""
    return value.split("\t----- bird -----\t", 1)[0].strip()


def execute_sql(conn, sql: str, timeout_seconds: int, max_rows: int) -> tuple[list[dict], str | None]:
    if not sql:
        return [], "empty_sql"
    try:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(f"SET statement_timeout = {int(timeout_seconds * 1000)}")
            cur.execute(sql)
            if cur.description is None:
                conn.rollback()
                return [], None
            rows = cur.fetchmany(max_rows)
            conn.rollback()
            return [dict(row) for row in rows], None
    except Exception as exc:
        try:
            conn.rollback()
        except Exception:
            pass
        return [], str(exc)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--db-host", default="localhost")
    parser.add_argument("--db-name", default="mapqa_socal")
    parser.add_argument("--db-user", default=os.environ.get("PGUSER", "postgres"))
    parser.add_argument("--db-password", default="")
    parser.add_argument("--db-port", type=int, default=5432)
    parser.add_argument("--timeout-seconds", type=int, default=360)
    parser.add_argument("--max-rows", type=int, default=100)
    args = parser.parse_args()

    dataset = {
        int(row["question_id"]): row
        for row in json.loads(args.dataset.read_text(encoding="utf-8"))
    }
    predictions = json.loads((args.run_dir / "-predictions.json").read_text(encoding="utf-8"))
    runtime = load_jsonl(args.run_dir / "-runtime_metrics.jsonl")

    conn = psycopg.connect(
        host=args.db_host,
        dbname=args.db_name,
        user=args.db_user,
        password=args.db_password,
        port=args.db_port,
    )

    exported = []
    try:
        for qid in sorted(dataset):
            item = dataset[qid]
            metrics = runtime.get(qid, {})
            sql = metrics.get("final_sql") or clean_prediction_sql(predictions.get(str(qid), ""))
            answers, exec_error = execute_sql(conn, sql, args.timeout_seconds, args.max_rows)
            final_exec_err = exec_error
            if final_exec_err is None:
                final_exec_err = metrics.get("final_sql_execution_error")

            exported.append({
                "question": item.get("question", ""),
                "sql": sql,
                "answers": answers,
                "metadata": {
                    "question_id": qid,
                    "db_id": item.get("db_id", "mapqa"),
                    "template_type": item.get("difficulty"),
                    "evidence": item.get("evidence", ""),
                    "pipeline_time_seconds": metrics.get("pipeline_time_s"),
                    "llm_latency_seconds": metrics.get("llm_latency_s"),
                    "prompt_tokens": metrics.get("prompt_tokens"),
                    "completion_tokens": metrics.get("completion_tokens"),
                    "total_tokens": metrics.get("total_tokens"),
                    "llm_calls": metrics.get("llm_calls"),
                    "final_exec_res": metrics.get("final_sql_eval_status"),
                    "final_exec_err": final_exec_err,
                    "final_sql_execution_status": metrics.get("final_sql_execution_status"),
                    "final_sql_execution_time_s": metrics.get("final_sql_execution_time_s"),
                    "final_sql_row_count": len(answers),
                },
            })
    finally:
        conn.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(exported, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {len(exported)} records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
