#!/usr/bin/env python3
import argparse
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Optional

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:
    psycopg = None
    dict_row = None


TABLE_BY_NAME_COLUMN = {
    "poi_name": "pois",
    "park_name": "parks",
    "road_name": "roads",
    "lake_name": "lakes",
    "region_name": "regions",
}


TYPE_HINT_BY_COLUMN = {
    "id": "INTEGER",
    "osm_id": "INTEGER",
    "geometry": "TEXT",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a compact CHESS osm.sqlite from rows stored inside GS-QA benchmark question.json files."
    )
    parser.add_argument("--benchmark-root", required=True, help="GS-QA benchmark directory containing T1...T28.")
    parser.add_argument("--out-sqlite", required=True)
    parser.add_argument("--include-splits", nargs="*", default=[], help="Optional split names, e.g. T1 T2. Default: all T*.")
    parser.add_argument("--hydrate-from-postgis", action="store_true", help="Look up each answer row in PostGIS and cache the matched DB row.")
    parser.add_argument("--include-gold-sql-results", action="store_true", help="Also execute each question.json gold SQL in PostGIS and cache returned rows.")
    parser.add_argument("--db-host", default="localhost")
    parser.add_argument("--db-name", default="gsqa")
    parser.add_argument("--db-user", default=os.environ.get("PGUSER", "postgres"))
    parser.add_argument("--db-password", default="")
    parser.add_argument("--db-port", type=int, default=5432)
    return parser.parse_args()


def split_sort_key(path: Path) -> tuple[int, str]:
    match = re.fullmatch(r"T(\d+)", path.name)
    return (int(match.group(1)) if match else 10**9, path.name)


def question_sort_key(path: Path) -> tuple[int, str]:
    name = path.parent.name.strip()
    return (int(name) if name.isdigit() else 10**9, str(path))


def infer_table(row: dict[str, Any], sql: str) -> Optional[str]:
    for column, table in TABLE_BY_NAME_COLUMN.items():
        if column in row:
            return table
    match = re.search(r"(?i)\bFROM\s+([A-Za-z_][A-Za-z0-9_]*)", sql or "")
    if match:
        table = match.group(1).lower()
        if table in set(TABLE_BY_NAME_COLUMN.values()):
            return table
    return None


def postgres_columns(conn: Any, table: str) -> list[str]:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = %s
            ORDER BY ordinal_position
            """,
            (table,),
        )
        return [row[0] for row in cursor.fetchall()]


def hydrate_row(conn: Any, table_columns: dict[str, list[str]], table: str, answer: dict[str, Any]) -> Optional[dict[str, Any]]:
    columns = table_columns[table]
    predicates = []
    params = []

    if "osm_id" in answer and "osm_id" in columns:
        predicates.append('"osm_id" = %s')
        params.append(answer["osm_id"])
    elif "id" in answer and "id" in columns:
        predicates.append('"id" = %s')
        params.append(answer["id"])
    else:
        for name_column in TABLE_BY_NAME_COLUMN:
            if name_column in answer and name_column in columns:
                predicates.append(f'"{name_column}" = %s')
                params.append(answer[name_column])
                break

    if not predicates:
        return None

    select_exprs = []
    for column in columns:
        safe_column = column.replace('"', '""')
        if column == "geometry":
            select_exprs.append(f'ST_AsText("{safe_column}"::geometry) AS "{safe_column}"')
        else:
            select_exprs.append(f'"{safe_column}"')
    sql = f'SELECT {", ".join(select_exprs)} FROM "{table}" WHERE {" AND ".join(predicates)} LIMIT 1'
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(sql, params)
        row = cursor.fetchone()
    return dict(row) if row else None


def batch_hydrate_rows(
    conn: Any,
    table_columns: dict[str, list[str]],
    table: str,
    objects: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    columns = table_columns[table]
    select_exprs = []
    for column in columns:
        safe_column = column.replace('"', '""')
        if column == "geometry":
            select_exprs.append(f'ST_AsText("{safe_column}"::geometry) AS "{safe_column}"')
        else:
            select_exprs.append(f'"{safe_column}"')

    queries: list[tuple[str, list[Any]]] = []
    osm_ids = sorted({obj.get("osm_id") for obj in objects if obj.get("osm_id") is not None})
    ids = sorted({obj.get("id") for obj in objects if obj.get("id") is not None})
    if osm_ids and "osm_id" in columns:
        queries.append((f'SELECT {", ".join(select_exprs)} FROM "{table}" WHERE "osm_id" = ANY(%s)', osm_ids))
    if ids and "id" in columns:
        queries.append((f'SELECT {", ".join(select_exprs)} FROM "{table}" WHERE "id" = ANY(%s)', ids))

    rows = []
    for sql, values in queries:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(sql, (values,))
            rows.extend(dict(row) for row in cursor.fetchall())
    conn.rollback()
    return rows


def strip_sql(sql: str) -> str:
    return sql.strip().rstrip(";")


def rows_from_gold_sql(conn: Any, sql: str, limit: int = 1000) -> list[dict[str, Any]]:
    if not sql.strip():
        return []
    try:
        with conn.cursor() as cursor:
            cursor.execute(f"SELECT * FROM ({strip_sql(sql)}) AS gsqa_gold LIMIT 0")
            columns = [desc.name for desc in cursor.description or []]
        if not columns:
            return []

        select_exprs = []
        for column in columns:
            safe_column = column.replace('"', '""')
            if column == "geometry":
                select_exprs.append(f'ST_AsText("{safe_column}"::geometry) AS "{safe_column}"')
            else:
                select_exprs.append(f'"{safe_column}"')
        wrapped_sql = f"SELECT {', '.join(select_exprs)} FROM ({strip_sql(sql)}) AS gsqa_gold LIMIT {int(limit)}"
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(wrapped_sql)
            rows = [dict(row) for row in cursor.fetchall()]
        conn.rollback()
        return rows
    except Exception:
        conn.rollback()
        return []


def sqlite_type(values: list[Any], column: str) -> str:
    if column in TYPE_HINT_BY_COLUMN:
        return TYPE_HINT_BY_COLUMN[column]
    non_null = [value for value in values if value is not None]
    if non_null and all(isinstance(value, int) and not isinstance(value, bool) for value in non_null):
        return "INTEGER"
    if non_null and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in non_null):
        return "REAL"
    return "TEXT"


def normalize_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True)
    return value


def extract_objects_with_geometry(data: Any, result: Optional[list[dict[str, Any]]] = None) -> list[dict[str, Any]]:
    if result is None:
        result = []
    if isinstance(data, dict):
        if "geometry" in data:
            result.append(data)
        for value in data.values():
            extract_objects_with_geometry(value, result)
    elif isinstance(data, list):
        for item in data:
            extract_objects_with_geometry(item, result)
    return result


def add_row(
    rows_by_table: dict[str, list[dict[str, Any]]],
    seen: dict[str, set[str]],
    table: str,
    row: dict[str, Any],
) -> bool:
    key = str(row.get("osm_id") or row.get("id") or json.dumps(row, sort_keys=True, default=str))
    if key in seen[table]:
        return False
    seen[table].add(key)
    rows_by_table[table].append({k: normalize_value(v) for k, v in row.items()})
    return True


def collect_rows(
    benchmark_root: Path,
    include_splits: set[str],
    pg_conn: Optional[Any] = None,
    include_gold_sql_results: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    rows_by_table: dict[str, list[dict[str, Any]]] = {table: [] for table in TABLE_BY_NAME_COLUMN.values()}
    seen: dict[str, set[str]] = {table: set() for table in TABLE_BY_NAME_COLUMN.values()}
    table_columns = {}
    if pg_conn is not None:
        table_columns = {table: postgres_columns(pg_conn, table) for table in TABLE_BY_NAME_COLUMN.values()}

    split_dirs = sorted(
        [p for p in benchmark_root.iterdir() if p.is_dir() and re.fullmatch(r"T\d+", p.name)],
        key=split_sort_key,
    )
    if include_splits:
        split_dirs = [p for p in split_dirs if p.name in include_splits]

    question_count = 0
    for split_dir in split_dirs:
        split_question_count = 0
        for question_path in sorted(split_dir.rglob("question.json"), key=question_sort_key):
            try:
                question = json.loads(question_path.read_text())
            except Exception:
                continue
            question_count += 1
            split_question_count += 1
            sql = question.get("sql", "")
            if include_gold_sql_results and pg_conn is not None:
                table = infer_table({}, sql)
                if table:
                    for row in rows_from_gold_sql(pg_conn, sql):
                        add_row(rows_by_table, seen, table, row)

            for obj in extract_objects_with_geometry(question):
                table = infer_table(obj, sql)
                if not table:
                    continue
                add_row(rows_by_table, seen, table, obj)
        print(
            f"scanned {split_dir.name}: {split_question_count} questions; "
            + ", ".join(f"{table}={len(rows_by_table[table])}" for table in rows_by_table),
            flush=True,
        )

    print(f"scanned total: {question_count} questions", flush=True)

    if pg_conn is not None:
        for table in list(rows_by_table):
            objects = rows_by_table[table]
            if not objects:
                print(f"{table}: 0 benchmark geometry objects", flush=True)
                continue
            print(f"{table}: hydrating {len(objects)} benchmark geometry objects from PostGIS", flush=True)
            hydrated_rows = batch_hydrate_rows(pg_conn, table_columns, table, objects)
            if hydrated_rows:
                rows_by_table[table] = []
                seen[table] = set()
                for row in hydrated_rows:
                    add_row(rows_by_table, seen, table, row)
            print(f"{table}: cached {len(rows_by_table[table])} rows", flush=True)
    return rows_by_table


def write_sqlite(out_sqlite: Path, rows_by_table: dict[str, list[dict[str, Any]]]) -> None:
    out_sqlite.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(out_sqlite)
    try:
        for table, rows in rows_by_table.items():
            conn.execute(f'DROP TABLE IF EXISTS "{table}"')
            if not rows:
                conn.execute(f'CREATE TABLE "{table}" (id INTEGER, geometry TEXT)')
                print(f"{table}: 0 rows")
                continue
            columns = sorted({column for row in rows for column in row.keys()})
            if "id" in columns:
                columns.remove("id")
                columns.insert(0, "id")
            if "geometry" in columns:
                columns.remove("geometry")
                columns.insert(1 if columns and columns[0] == "id" else 0, "geometry")

            column_defs = []
            for column in columns:
                values = [row.get(column) for row in rows]
                column_defs.append(f'"{column}" {sqlite_type(values, column)}')
            conn.execute(f'CREATE TABLE "{table}" ({", ".join(column_defs)})')

            placeholders = ", ".join(["?"] * len(columns))
            quoted_columns = ", ".join(f'"{column}"' for column in columns)
            insert_sql = f'INSERT INTO "{table}" ({quoted_columns}) VALUES ({placeholders})'
            conn.executemany(insert_sql, [tuple(row.get(column) for column in columns) for row in rows])
            conn.commit()
            print(f"{table}: wrote {len(rows)} rows", flush=True)
    finally:
        conn.close()


def main() -> None:
    args = parse_args()
    pg_conn = None
    if args.hydrate_from_postgis:
        if psycopg is None:
            raise ImportError("Use the CHESS conda env with psycopg installed to hydrate from PostGIS.")
        pg_conn = psycopg.connect(
            host=args.db_host,
            dbname=args.db_name,
            user=args.db_user,
            password=args.db_password,
            port=args.db_port,
        )
    try:
        rows_by_table = collect_rows(
            Path(args.benchmark_root),
            set(args.include_splits),
            pg_conn,
            include_gold_sql_results=args.include_gold_sql_results,
        )
        write_sqlite(Path(args.out_sqlite), rows_by_table)
    finally:
        if pg_conn is not None:
            pg_conn.close()
    print(f"wrote {args.out_sqlite}")


if __name__ == "__main__":
    main()
