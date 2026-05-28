#!/usr/bin/env python3
import argparse
import sqlite3
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row


DEFAULT_TABLES = ["pois", "lakes", "parks", "roads", "regions"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mirror selected PostGIS tables into CHESS osm.sqlite for preprocessing.")
    parser.add_argument("--out-sqlite", required=True)
    parser.add_argument("--tables", nargs="+", default=DEFAULT_TABLES)
    parser.add_argument("--row-limit", type=int, default=0, help="0 means export all rows.")
    parser.add_argument("--db-host", default="localhost")
    parser.add_argument("--db-name", default="gsqa")
    parser.add_argument("--db-user", required=True)
    parser.add_argument("--db-password", default="")
    parser.add_argument("--db-port", type=int, default=5432)
    return parser.parse_args()


def postgres_columns(conn: psycopg.Connection, table: str) -> list[dict[str, Any]]:
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(
            """
            SELECT column_name, data_type, udt_name
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = %s
            ORDER BY ordinal_position
            """,
            (table,),
        )
        return [dict(row) for row in cursor.fetchall()]


def sqlite_type(column: dict[str, Any]) -> str:
    name = column["column_name"].lower()
    data_type = column["data_type"]
    if name == "geometry":
        return "TEXT"
    if data_type in {"integer", "bigint", "smallint"}:
        return "INTEGER"
    if data_type in {"numeric", "double precision", "real"}:
        return "REAL"
    return "TEXT"


def create_sqlite_table(sqlite_conn: sqlite3.Connection, table: str, columns: list[dict[str, Any]]) -> None:
    column_defs = [f'"{column["column_name"]}" {sqlite_type(column)}' for column in columns]
    sqlite_conn.execute(f'DROP TABLE IF EXISTS "{table}"')
    sqlite_conn.execute(f'CREATE TABLE "{table}" ({", ".join(column_defs)})')


def export_table(
    pg_conn: psycopg.Connection,
    sqlite_conn: sqlite3.Connection,
    table: str,
    columns: list[dict[str, Any]],
    row_limit: int,
) -> int:
    select_exprs = []
    column_names = [column["column_name"] for column in columns]
    for column in columns:
        name = column["column_name"]
        safe_name = name.replace('"', '""')
        if name == "geometry" and column["udt_name"] in {"geometry", "geography"}:
            select_exprs.append(f'ST_AsText("{safe_name}"::geometry) AS "{safe_name}"')
        else:
            select_exprs.append(f'"{safe_name}"')

    limit_sql = f" LIMIT {int(row_limit)}" if row_limit > 0 else ""
    query = f'SELECT {", ".join(select_exprs)} FROM "{table}"{limit_sql}'
    placeholders = ", ".join(["?"] * len(column_names))
    quoted_column_names = ", ".join(f'"{column}"' for column in column_names)
    insert_sql = f'INSERT INTO "{table}" ({quoted_column_names}) VALUES ({placeholders})'

    count = 0
    with pg_conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute(query)
        while True:
            rows = cursor.fetchmany(1000)
            if not rows:
                break
            values = [tuple(row.get(column) for column in column_names) for row in rows]
            sqlite_conn.executemany(insert_sql, values)
            sqlite_conn.commit()
            count += len(rows)
            print(f"{table}: exported {count} rows")
    return count


def main() -> None:
    args = parse_args()
    out_sqlite = Path(args.out_sqlite)
    out_sqlite.parent.mkdir(parents=True, exist_ok=True)

    pg_conn = psycopg.connect(
        host=args.db_host,
        dbname=args.db_name,
        user=args.db_user,
        password=args.db_password,
        port=args.db_port,
    )
    sqlite_conn = sqlite3.connect(out_sqlite)
    try:
        for table in args.tables:
            columns = postgres_columns(pg_conn, table)
            if not columns:
                print(f"{table}: missing in PostGIS, skipped")
                continue
            create_sqlite_table(sqlite_conn, table, columns)
            count = export_table(pg_conn, sqlite_conn, table, columns, args.row_limit)
            print(f"{table}: done, rows={count}")
    finally:
        sqlite_conn.close()
        pg_conn.close()

    print(f"wrote {out_sqlite}")


if __name__ == "__main__":
    main()
