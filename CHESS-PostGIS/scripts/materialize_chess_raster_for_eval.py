#!/usr/bin/env python3
import argparse
import csv
import json
import re
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row


RUNS = {
    "raster_only": (
        "data/dev/gsqa_raster_only_postgis.json",
        "results/dev/CHESS_IR_CG_UT_GEMINI_RASTER/gsqa_raster_only_postgis",
    ),
    "raster_vector": (
        "data/dev/gsqa_raster_vector_postgis.json",
        "results/dev/CHESS_IR_CG_UT_GEMINI_RASTER/gsqa_raster_vector_postgis",
    ),
    "extended": (
        "data/dev/gsqa_extended_postgis.json",
        "results/dev/CHESS_IR_CG_UT_GEMINI_RASTER/gsqa_extended_postgis",
    ),
}


def clean_sql(sql: str) -> str:
    sql = re.sub(r"</?FINAL_ANSWER>", " ", sql or "")
    sql = sql.replace("```sql", "").replace("```", "")
    sql = re.sub(r"(['\"])3DDDA\1", "'32BF'", sql, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", sql).replace('"', "'").strip("`.; ") + ";"


def parse_evidence(evidence: str) -> tuple[str, str]:
    source = ""
    answer_type = ""
    for part in (evidence or "").split(";"):
        part = part.strip()
        if part.startswith("source="):
            source = part.split("=", 1)[1].strip()
        elif part.startswith("answer_type="):
            answer_type = part.split("=", 1)[1].strip()
    return source, answer_type


def timing_template_name(source_stem: str) -> str:
    aliases = {
        "intersects_area_max+name+max_elevation": "intersects:area_max+name+max_elevation",
        "intersects_area_total+area+avg_slope": "intersects:area_total+area+avg_slope",
        "intersects_length_max+name+slope": "intersects:length_max+name+slope",
        "intersects_length_total+length+slope": "intersects:length_total+length+slope",
    }
    return aliases.get(source_stem, source_stem)


def load_timing_limits(path: Path | None) -> dict[tuple[str, int], float]:
    if not path:
        return {}
    limits: dict[tuple[str, int], float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            template = (row.get("template_name") or "").strip()
            if not template:
                continue
            try:
                question_index = int(row.get("question_index") or "")
                seconds = float(row.get("sql_execution_seconds") or "")
            except ValueError:
                continue
            if seconds <= 0:
                continue
            key = (template, question_index)
            if key not in limits or seconds > limits[key]:
                limits[key] = seconds
    return limits


def timing_limit_for(
    timing_limits: dict[tuple[str, int], float],
    source_stem: str,
    source_index_zero_based: int,
    multiplier: float,
    minimum_seconds: float,
) -> float | None:
    if not timing_limits:
        return None
    template = timing_template_name(source_stem)
    for question_index in (source_index_zero_based + 1, source_index_zero_based):
        value = timing_limits.get((template, question_index))
        if value is not None:
            return max(value * multiplier, minimum_seconds)
    return None


def latest_run_dir(root: Path) -> Path:
    dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not dirs:
        raise FileNotFoundError(f"No run directories under {root}")
    return dirs[-1]


def jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return value.hex()
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def execute(conn: Any, sql: str, timeout_seconds: float | None) -> dict[str, Any]:
    if not sql.strip():
        return {"error": "missing SQL", "output": [], "execution_seconds": 0.0, "timeout_seconds": timeout_seconds}
    started = time.perf_counter()
    try:
        with conn.cursor(row_factory=dict_row) as cur:
            if timeout_seconds is not None and timeout_seconds > 0:
                cur.execute(f"SET LOCAL statement_timeout = {max(1, int(timeout_seconds * 1000))}")
            else:
                cur.execute("SET LOCAL statement_timeout = 0")
            cur.execute(clean_sql(sql))
            rows = cur.fetchall() if cur.description is not None else []
        conn.rollback()
        elapsed = time.perf_counter() - started
        return {
            "error": "",
            "output": [jsonable(dict(row)) for row in rows],
            "execution_seconds": elapsed,
            "timeout_seconds": timeout_seconds,
        }
    except Exception as exc:
        conn.rollback()
        elapsed = time.perf_counter() - started
        message = str(exc)
        if "statement timeout" in message.lower():
            message = f"timeout: exceeded ground-truth SQL time budget ({timeout_seconds:.4f}s)"
        return {
            "error": message,
            "output": [],
            "execution_seconds": elapsed,
            "timeout_seconds": timeout_seconds,
        }


def load_gold_cache(cache_root: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    gold: dict[tuple[str, str, str], dict[str, Any]] = {}
    for query_type in RUNS:
        for path in (cache_root / f"gemini_{query_type}").glob("*/*.json"):
            if path.name in {"summary.json", "run_summary.json"}:
                continue
            try:
                record = json.loads(path.read_text())
            except Exception:
                continue
            key = (
                query_type,
                record.get("source_stem", ""),
                re.sub(r"\s+", " ", record.get("question", "")).strip(),
            )
            gold[key] = {
                "gold_exec": record.get("gold_exec", {"error": "", "output": []}),
                "gold_answers": record.get("gold_answers", []),
                "gold_sql": record.get("gold_sql", ""),
            }
    return gold


def extract_final_sql(history: list[dict[str, Any]]) -> str:
    for step in reversed(history):
        if step.get("tool_name") == "runtime_metrics" and step.get("final_sql"):
            return step["final_sql"]
        if "final_SQL" in step and isinstance(step["final_SQL"], dict):
            return step["final_SQL"].get("PREDICTED_SQL") or step["final_SQL"].get("SQL") or ""
        if "final_sql" in step and isinstance(step["final_sql"], dict):
            return step["final_sql"].get("SQL") or step["final_sql"].get("PREDICTED_SQL") or ""
    return ""


def materialize_one(
    repo: Path,
    conn: Any,
    query_type: str,
    dataset_path: Path,
    run_root: Path,
    out_root: Path,
    gold_cache: dict[tuple[str, str, str], dict[str, Any]],
    default_timeout_seconds: int,
    timing_limits: dict[tuple[str, int], float],
    timing_timeout_multiplier: float,
    min_timing_timeout_seconds: float,
    retry_errors: bool,
    overwrite: bool,
) -> tuple[int, int]:
    dataset = json.loads(dataset_path.read_text())
    run_dir = latest_run_dir(run_root)
    written = 0
    executed = 0
    source_seen: dict[str, int] = {}
    for item in dataset:
        qid = item["question_id"]
        history_path = run_dir / f"{qid}_osm.json"
        if not history_path.exists():
            history_path = run_dir / f"{qid}_{item.get('db_id', 'osm')}.json"
        history = json.loads(history_path.read_text()) if history_path.exists() else []
        predicted_sql = extract_final_sql(history)
        source_file, answer_type = parse_evidence(item.get("evidence", ""))
        source_stem = Path(source_file).stem if source_file else "unknown"
        source_index = source_seen.get(source_stem, 0)
        source_seen[source_stem] = source_index + 1
        record_id = f"{source_stem}-{qid}"
        out_dir = out_root / f"chess_{query_type}" / source_stem
        out_path = out_dir / f"{source_stem}-{qid}.json"
        if out_path.exists() and not overwrite:
            if not retry_errors:
                continue
            try:
                existing = json.loads(out_path.read_text())
                if not (existing.get("predicted_exec") or {}).get("error"):
                    continue
            except Exception:
                pass
        gold_key = (query_type, source_stem, re.sub(r"\s+", " ", item.get("question", "")).strip())
        gold_record = gold_cache.get(gold_key, {})
        timing_timeout = timing_limit_for(
            timing_limits,
            source_stem,
            source_index,
            timing_timeout_multiplier,
            min_timing_timeout_seconds,
        )
        timeout_seconds = timing_timeout if timing_timeout is not None else default_timeout_seconds

        predicted_exec = execute(conn, predicted_sql, timeout_seconds) if predicted_sql else {"error": "missing SQL", "output": []}
        executed += int(not bool(predicted_exec.get("error")))

        record = {
            "id": record_id,
            "query_type": query_type,
            "source_file": source_file,
            "source_stem": source_stem,
            "answer_type": answer_type,
            "question": item.get("question", ""),
            "gold_sql": gold_record.get("gold_sql") or item.get("SQL", ""),
            "predicted_sql": predicted_sql,
            "predicted_sql_candidates": [predicted_sql] if predicted_sql else [],
            "model_output": predicted_sql,
            "generation_error": "" if predicted_sql else "missing final SQL",
            "gold_answers": gold_record.get("gold_answers", []),
            "gold_exec": gold_record.get("gold_exec", {"error": "missing gold cache record", "output": []}),
            "predicted_exec": predicted_exec,
            "gold_sql_timeout_seconds": timing_timeout,
            "source_index": source_index,
            "chess_history_file": str(history_path),
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        written += 1
    return written, executed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".", help="CHESS repo root")
    parser.add_argument("--output-root", default="../GS-QA-experiment/GS-QA/baselines/cache")
    parser.add_argument("--gold-cache-root", default="../GS-QA-experiment/GS-QA/baselines/cache")
    parser.add_argument("--timeout-seconds", type=int, default=360)
    parser.add_argument("--timing-csv", default="", help="Per-question ground-truth timing CSV. If set, each CHESS SQL uses the matching gold SQL time as timeout.")
    parser.add_argument("--timing-timeout-multiplier", type=float, default=1.0)
    parser.add_argument("--min-timing-timeout-seconds", type=float, default=0.0)
    parser.add_argument("--retry-errors", action="store_true", help="Re-execute existing records whose predicted_exec has an error.")
    parser.add_argument("--overwrite", action="store_true", help="Re-execute and overwrite every record.")
    parser.add_argument("--pg-host", default="localhost")
    parser.add_argument("--pg-database", default="gsqa")
    parser.add_argument("--pg-user", default="zshan011")
    parser.add_argument("--pg-password", default="")
    parser.add_argument("--pg-port", type=int, default=5432)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    out_root = Path(args.output_root).expanduser().resolve()
    gold_cache = load_gold_cache(Path(args.gold_cache_root).expanduser().resolve())
    timing_csv = Path(args.timing_csv).expanduser().resolve() if args.timing_csv else None
    timing_limits = load_timing_limits(timing_csv)
    print(f"loaded gold cache records={len(gold_cache)}")
    print(f"loaded timing limits={len(timing_limits)}")
    conn = psycopg.connect(
        host=args.pg_host,
        dbname=args.pg_database,
        user=args.pg_user,
        password=args.pg_password,
        port=args.pg_port,
    )
    try:
        total_written = 0
        total_executed = 0
        for query_type, (dataset_rel, run_rel) in RUNS.items():
            written, executed = materialize_one(
                repo,
                conn,
                query_type,
                repo / dataset_rel,
                repo / run_rel,
                out_root,
                gold_cache,
                args.timeout_seconds,
                timing_limits,
                args.timing_timeout_multiplier,
                args.min_timing_timeout_seconds,
                args.retry_errors,
                args.overwrite,
            )
            print(f"{query_type}: wrote={written} predicted_exec_success={executed}")
            total_written += written
            total_executed += executed
        print(f"total: wrote={total_written} predicted_exec_success={total_executed}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
