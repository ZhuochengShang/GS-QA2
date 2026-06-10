#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from baselines.chess.collect_chess_results import (
    answer_text_from_records,
    connect_postgres,
    enrich_records_by_name,
    execute_sql,
    extract_sql_candidate,
)


def load_json_list(path: Path):
    if not path.exists():
        return []
    return json.loads(path.read_text())


def load_question_text(question_root: Path, qid: int) -> str:
    question_path = question_root / f"{qid:3d}" / "question.json"
    if not question_path.exists():
        return ""
    return json.loads(question_path.read_text()).get("question", "")


def flatten_cached_outputs(exec_record: dict):
    records = []
    errors = []
    for block in exec_record.get("records", []) or []:
        if not isinstance(block, dict):
            continue
        err = block.get("error")
        if err:
            errors.append(str(err))
        out = block.get("output") or []
        if isinstance(out, list):
            records.extend(out)
    return records, "\n".join(errors)


def usage_from_generate(generate_record: dict):
    usage = generate_record.get("usage_metadata") or {}
    return {
        "prompt_tokens": int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("output_tokens") or usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cache_dir", type=Path, help="Text2SQL cache dir, e.g. baselines/cache/gemini/T13")
    parser.add_argument("--execute", action="store_true", help="Re-execute generated SQL instead of using sql_exec.json cache.")
    parser.add_argument("--enrich-name-results", action="store_true")
    parser.add_argument("--pg-host", default="localhost")
    parser.add_argument("--pg-database", default="gsqa")
    parser.add_argument("--pg-user", default="")
    parser.add_argument("--pg-password", default="")
    parser.add_argument("--pg-port", type=int, default=5432)
    parser.add_argument("--statement-timeout-seconds", type=float, default=360.0)
    parser.add_argument("--write-question-results", action="store_true")
    parser.add_argument("--question-root", type=Path, required=True)
    parser.add_argument("--question-output-root", type=Path, required=True)
    args = parser.parse_args()

    sql_generate = load_json_list(args.cache_dir / "sql_generate.json")
    sql_exec = load_json_list(args.cache_dir / "sql_exec.json")
    exec_by_id = {record.get("id"): record for record in sql_exec}
    question_dirs = sorted(
        [p for p in args.question_root.iterdir() if p.is_dir() and p.name.strip().isdigit()],
        key=lambda p: int(p.name),
    )
    if len(question_dirs) != len(sql_generate):
        raise ValueError(
            f"question/cache count mismatch: {len(question_dirs)} questions vs "
            f"{len(sql_generate)} sql_generate records"
        )

    conn = None
    if args.execute or args.enrich_name_results:
        conn = connect_postgres(
            {
                "host": args.pg_host,
                "dbname": args.pg_database,
                "user": args.pg_user,
                "password": args.pg_password,
                "port": args.pg_port,
            }
        )

    rows = []
    try:
        for qdir, gen in zip(question_dirs, sql_generate):
            qid = int(qdir.name)
            sql = extract_sql_candidate(gen.get("content", ""))
            final_exec_err = ""
            if args.execute:
                try:
                    records = execute_sql(conn, sql, args.statement_timeout_seconds)
                    conn.commit()
                except Exception as exc:
                    conn.rollback()
                    records = []
                    final_exec_err = str(exc)
            else:
                records, final_exec_err = flatten_cached_outputs(exec_by_id.get(gen.get("id"), {}))

            if args.enrich_name_results and records:
                try:
                    records = enrich_records_by_name(conn, records)
                    conn.commit()
                except Exception as exc:
                    conn.rollback()
                    final_exec_err = (final_exec_err + "\n" if final_exec_err else "") + f"ENRICH_ERROR: {exc}"

            usage = usage_from_generate(gen)
            payload = {
                "question": load_question_text(args.question_root, qid),
                "sql": sql,
                "answers": records,
                "metadata": {
                    "question_id": qid,
                    "cache_id": gen.get("id"),
                    "db_id": "gsqa",
                    "pipeline_time_seconds": gen.get("run_seconds", 0),
                    "llm_latency_seconds": gen.get("run_seconds", 0),
                    "prompt_tokens": usage["prompt_tokens"],
                    "completion_tokens": usage["completion_tokens"],
                    "total_tokens": usage["total_tokens"],
                    "llm_calls": 1 if gen.get("content") else 0,
                    "final_exec_res": "",
                    "final_exec_err": final_exec_err,
                    "answer_text": answer_text_from_records(records),
                },
            }
            if args.write_question_results:
                out_dir = args.question_output_root / f"{qid:3d}"
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "question.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
            rows.append(payload)
    finally:
        if conn is not None:
            conn.close()

    args.question_output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.question_output_root / "collected_text2sql_results.json"
    summary_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"cache_dir: {args.cache_dir}")
    print(f"wrote: {len(rows)}")
    print(f"question_results: {args.question_output_root}")
    print(f"summary_json: {summary_path}")


if __name__ == "__main__":
    main()
