#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row


def load_predictions(path: Path) -> dict[int, str]:
    raw = json.loads(path.read_text())
    predictions: dict[int, str] = {}
    for key, value in raw.items():
        if not isinstance(value, str):
            continue
        sql = value.split("\t----- bird -----\t")[0].strip()
        if sql:
            predictions[int(key)] = sql
    return predictions


def normalize_value(value: Any) -> Any:
    if isinstance(value, memoryview):
        return value.tobytes().hex()
    if isinstance(value, bytes):
        return value.hex()
    return value


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: normalize_value(value) for key, value in row.items()}


def strip_sql(sql: str) -> str:
    return sql.strip().rstrip(";")


def geometry_columns(conn: psycopg.Connection, sql: str) -> set[str]:
    probe_sql = f"SELECT * FROM ({strip_sql(sql)}) AS chess_pred LIMIT 0"
    with conn.cursor() as cursor:
        cursor.execute(probe_sql)
        columns = [desc.name for desc in cursor.description or []]

    geom_cols: set[str] = set()
    for column in columns:
        safe_col = column.replace('"', '""')
        type_sql = f'SELECT pg_typeof("{safe_col}")::text FROM ({strip_sql(sql)}) AS chess_pred WHERE "{safe_col}" IS NOT NULL LIMIT 1'
        try:
            with conn.cursor() as cursor:
                cursor.execute(type_sql)
                row = cursor.fetchone()
            if row and row[0] in {"geometry", "geography"}:
                geom_cols.add(column)
        except Exception:
            conn.rollback()
    return geom_cols


def execute_prediction(conn: psycopg.Connection, sql: str, limit: int) -> tuple[list[dict[str, Any]], str]:
    try:
        geom_cols = geometry_columns(conn, sql)
        with conn.cursor() as cursor:
            cursor.execute(f"SELECT * FROM ({strip_sql(sql)}) AS chess_pred LIMIT 0")
            columns = [desc.name for desc in cursor.description or []]

        select_exprs = []
        for column in columns:
            safe_col = column.replace('"', '""')
            if column in geom_cols:
                select_exprs.append(f'ST_AsText("{safe_col}"::geometry) AS "{safe_col}"')
            else:
                select_exprs.append(f'"{safe_col}"')

        wrapped_sql = f"SELECT {', '.join(select_exprs)} FROM ({strip_sql(sql)}) AS chess_pred LIMIT {int(limit)}"
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(wrapped_sql)
            rows = [normalize_row(dict(row)) for row in cursor.fetchall()]
        conn.rollback()
        return rows, ""
    except Exception as exc:
        conn.rollback()
        return [], str(exc)


def question_json_path(benchmark_split_dir: Path, question_id: int) -> Path | None:
    direct = benchmark_split_dir / str(question_id) / "question.json"
    if direct.exists():
        return direct
    matches = sorted(benchmark_split_dir.glob(f"*{question_id}/question.json"))
    return matches[0] if matches else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute CHESS PostGIS predictions and export GS-QA-shaped answers.")
    parser.add_argument("--predictions", required=True, help="CHESS -predictions.json")
    parser.add_argument("--dataset", required=True, help="CHESS GS-QA dataset JSON for this T split")
    parser.add_argument("--benchmark-split-dir", required=True, help="Original GS-QA benchmark/TN directory")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--write-sidecars", action="store_true", help="Also write chess_prediction.json beside each question.json")
    parser.add_argument("--db-host", default="localhost")
    parser.add_argument("--db-name", default="gsqa")
    parser.add_argument("--db-user", default="postgres")
    parser.add_argument("--db-password", default="")
    parser.add_argument("--db-port", type=int, default=5432)
    args = parser.parse_args()

    predictions = load_predictions(Path(args.predictions))
    dataset = json.loads(Path(args.dataset).read_text())
    benchmark_split_dir = Path(args.benchmark_split_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = psycopg.connect(
        host=args.db_host,
        dbname=args.db_name,
        user=args.db_user,
        password=args.db_password,
        port=args.db_port,
    )

    outputs = []
    try:
        for item in dataset:
            question_id = int(item["question_id"])
            sql = predictions.get(question_id, "")
            rows, error = execute_prediction(conn, sql, args.limit) if sql else ([], "missing_prediction")
            source_question_path = question_json_path(benchmark_split_dir, question_id)
            question = item.get("question", "")
            if source_question_path:
                try:
                    question = json.loads(source_question_path.read_text()).get("question", question)
                except Exception:
                    pass

            output = {
                "question_id": question_id,
                "source_question_json": str(source_question_path) if source_question_path else "",
                "question": question,
                "predicted_sql": sql,
                "answers": rows,
                "status": "ok" if not error else "error",
                "error": error,
            }
            outputs.append(output)

            if args.write_sidecars and source_question_path:
                sidecar = source_question_path.parent / "chess_prediction.json"
                sidecar.write_text(json.dumps(output, indent=2, ensure_ascii=True))
    finally:
        conn.close()

    combined = out_dir / "chess_gsqa_answers.json"
    combined.write_text(json.dumps(outputs, indent=2, ensure_ascii=True))
    print(f"wrote {combined}")


if __name__ == "__main__":
    main()
