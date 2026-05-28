#!/usr/bin/env python3
import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import tuple_row


@dataclass
class EvalRow:
    question_id: int
    db_id: str
    pred_ok: bool
    gold_ok: bool
    exact_match: bool
    pred_error: str = ""
    gold_error: str = ""
    pred_sql: str = ""
    gold_sql: str = ""


def load_chess_predictions(path: Path) -> dict[int, str]:
    raw = json.loads(path.read_text())
    out: dict[int, str] = {}
    for k, v in raw.items():
        qid = int(k)
        if not isinstance(v, str):
            continue
        # CHESS format: "<sql>\t----- bird -----\t<db_id>"
        sql = v.split("\t----- bird -----\t")[0].strip()
        if sql:
            out[qid] = sql
    return out


def normalize_value(v: Any) -> Any:
    # Normalize Postgres driver return types for stable comparison
    if isinstance(v, memoryview):
        return v.tobytes().hex()
    if isinstance(v, bytes):
        return v.hex()
    if isinstance(v, float):
        return round(v, 8)
    return v


def normalize_rows(rows: list[tuple]) -> list[tuple]:
    norm = [tuple(normalize_value(c) for c in r) for r in rows]
    # Sort by repr for order-insensitive equivalence on SELECTs without ORDER BY
    return sorted(norm, key=repr)


def run_sql(conn: psycopg.Connection, sql: str, timeout_ms: int) -> tuple[bool, list[tuple], str]:
    try:
        with conn.cursor(row_factory=tuple_row) as cur:
            cur.execute(f"SET statement_timeout = {int(timeout_ms)}")
            cur.execute(sql)
            rows = cur.fetchall()
        conn.rollback()
        return True, rows, ""
    except Exception as e:
        conn.rollback()
        return False, [], str(e)


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate CHESS SQL vs GS-QA gold SQL on PostGIS by execution equivalence")
    p.add_argument("--predictions", required=True, help="Path to CHESS -predictions.json")
    p.add_argument("--dataset", required=True, help="Path to GS-QA dataset JSON used by CHESS (with SQL field)")
    p.add_argument("--out-csv", default="postgis_exec_eval.csv", help="Output CSV path")
    p.add_argument("--out-json", default="postgis_exec_eval_summary.json", help="Output JSON summary path")

    p.add_argument("--db-host", default="localhost")
    p.add_argument("--db-name", default="gsqa")
    p.add_argument("--db-user", default="postgres")
    p.add_argument("--db-password", default="postgres")
    p.add_argument("--db-port", type=int, default=5432)
    p.add_argument("--timeout-ms", type=int, default=120000)
    args = p.parse_args()

    preds = load_chess_predictions(Path(args.predictions))
    dataset = json.loads(Path(args.dataset).read_text())

    conn = psycopg.connect(
        host=args.db_host,
        dbname=args.db_name,
        user=args.db_user,
        password=args.db_password,
        port=args.db_port,
    )

    rows: list[EvalRow] = []
    for item in dataset:
        qid = int(item["question_id"])
        db_id = str(item.get("db_id", ""))
        gold_sql = str(item.get("SQL", "")).strip()
        pred_sql = preds.get(qid, "").strip()

        if not gold_sql:
            rows.append(EvalRow(qid, db_id, bool(pred_sql), False, False, gold_error="missing_gold_sql", pred_sql=pred_sql, gold_sql=gold_sql))
            continue
        if not pred_sql:
            rows.append(EvalRow(qid, db_id, False, True, False, pred_error="missing_pred_sql", pred_sql=pred_sql, gold_sql=gold_sql))
            continue

        pred_ok, pred_rows, pred_err = run_sql(conn, pred_sql, args.timeout_ms)
        gold_ok, gold_rows, gold_err = run_sql(conn, gold_sql, args.timeout_ms)

        exact = False
        if pred_ok and gold_ok:
            exact = normalize_rows(pred_rows) == normalize_rows(gold_rows)

        rows.append(
            EvalRow(
                question_id=qid,
                db_id=db_id,
                pred_ok=pred_ok,
                gold_ok=gold_ok,
                exact_match=exact,
                pred_error=pred_err,
                gold_error=gold_err,
                pred_sql=pred_sql,
                gold_sql=gold_sql,
            )
        )

    conn.close()

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "question_id",
            "db_id",
            "pred_ok",
            "gold_ok",
            "exact_match",
            "pred_error",
            "gold_error",
        ])
        for r in rows:
            w.writerow([
                r.question_id,
                r.db_id,
                int(r.pred_ok),
                int(r.gold_ok),
                int(r.exact_match),
                r.pred_error,
                r.gold_error,
            ])

    total = len(rows)
    both_ok = sum(1 for r in rows if r.pred_ok and r.gold_ok)
    exact = sum(1 for r in rows if r.exact_match)
    pred_errors = sum(1 for r in rows if not r.pred_ok)
    gold_errors = sum(1 for r in rows if not r.gold_ok)

    summary = {
        "total": total,
        "both_executed": both_ok,
        "exact_match_count": exact,
        "exact_match_rate_over_total": (exact / total) if total else 0.0,
        "exact_match_rate_over_both_executed": (exact / both_ok) if both_ok else 0.0,
        "pred_error_count": pred_errors,
        "gold_error_count": gold_errors,
        "csv": str(out_csv),
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
