#!/usr/bin/env python3
"""
Load GS-QA OSM GeoJSON extracts into a PostGIS table.

This is the data-loading step before running CHESS in postgis mode. CHESS itself
executes SQL against PostGIS; it does not read GeoJSON files during inference.
"""

import argparse
import glob
import json
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import psycopg
from shapely.geometry import shape


SQL_TYPE_BY_SCHEMA_TYPE = {
    "string": "VARCHAR(255)",
    "integer": "BIGINT",
    "float": "DOUBLE PRECISION",
}

NAME_COLUMN_BY_TABLE = {
    "pois": "poi_name",
    "parks": "park_name",
    "roads": "road_name",
    "lakes": "lake_name",
    "regions": "region_name",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load one GS-QA OSM GeoJSON glob into a PostGIS table."
    )
    parser.add_argument("--geojson-glob", required=True, help="Quoted glob, e.g. '.../poi/*.geojson'.")
    parser.add_argument("--table", required=True, choices=sorted(NAME_COLUMN_BY_TABLE))
    parser.add_argument("--schema-json", required=True, help="GS-QA schema JSON for the target table.")
    parser.add_argument("--append", action="store_true", help="Append rows instead of dropping/recreating the table.")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--db-host", default="localhost")
    parser.add_argument("--db-name", required=True)
    parser.add_argument("--db-user", required=True)
    parser.add_argument("--db-password", default="")
    parser.add_argument("--db-port", type=int, default=5432)
    return parser.parse_args()


def load_schema(path: str) -> Dict[str, Any]:
    with open(path, "r") as file:
        schema = json.load(file)
    if "geometry" not in schema:
        raise ValueError(f"{path} does not contain a geometry field.")
    return schema


def iter_features(paths: Sequence[str]) -> Iterable[Dict[str, Any]]:
    for path in paths:
        with open(path, "r") as file:
            data = json.load(file)
        for feature in data.get("features", []):
            if feature.get("geometry"):
                yield feature


def normalize_tags(feature: Dict[str, Any], table: str) -> Dict[str, Any]:
    properties = feature.get("properties") or {}
    tags = dict(properties.get("tagsMap") or properties)

    name_column = NAME_COLUMN_BY_TABLE[table]
    if "name" in tags and name_column not in tags:
        tags[name_column] = tags["name"]
    if "id" in properties and "osm_id" not in tags:
        tags["osm_id"] = properties["id"]

    for key in list(tags):
        if key.startswith("addr:"):
            tags[key.replace(":", "_")] = tags[key]
    return tags


def create_table_sql(table: str, schema: Dict[str, Any]) -> str:
    columns = []
    for column, schema_type in schema.items():
        if column == "geometry":
            continue
        sql_type = SQL_TYPE_BY_SCHEMA_TYPE.get(schema_type)
        if sql_type is None:
            raise ValueError(f"Unsupported schema type for {column}: {schema_type}")
        columns.append(f'"{column}" {sql_type}')
    column_sql = ",\n        ".join(columns)
    return f"""
DROP TABLE IF EXISTS "{table}" CASCADE;
CREATE TABLE "{table}" (
    id SERIAL PRIMARY KEY,
    geometry GEOGRAPHY(GEOMETRY, 4326),
        {column_sql}
);
CREATE INDEX "{table}_geometry_gix" ON "{table}" USING GIST (geometry);
"""


def rows_for_batch(
    features: Sequence[Dict[str, Any]],
    table: str,
    schema: Dict[str, Any],
) -> List[Tuple[Any, ...]]:
    columns = [column for column in schema if column != "geometry"]
    rows = []
    for feature in features:
        tags = normalize_tags(feature, table)
        geometry_wkt = shape(feature["geometry"]).wkt
        rows.append((geometry_wkt, *[tags.get(column) for column in columns]))
    return rows


def insert_batch(conn: psycopg.Connection, table: str, schema: Dict[str, Any], features: Sequence[Dict[str, Any]]) -> int:
    if not features:
        return 0
    columns = [column for column in schema if column != "geometry"]
    quoted_columns = ['"geometry"', *[f'"{column}"' for column in columns]]
    placeholders = ["ST_GeogFromText(%s)", *["%s" for _ in columns]]
    sql = f"""
INSERT INTO "{table}" ({", ".join(quoted_columns)})
VALUES ({", ".join(placeholders)})
"""
    rows = rows_for_batch(features, table, schema)
    with conn.cursor() as cursor:
        cursor.executemany(sql, rows)
    return len(rows)


def main() -> None:
    args = parse_args()
    paths = sorted(glob.glob(args.geojson_glob))
    if not paths:
        raise FileNotFoundError(f"No GeoJSON files matched: {args.geojson_glob}")

    schema = load_schema(args.schema_json)
    conn = psycopg.connect(
        host=args.db_host,
        dbname=args.db_name,
        user=args.db_user,
        password=args.db_password,
        port=args.db_port,
    )
    try:
        with conn.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
            if not args.append:
                cursor.execute(create_table_sql(args.table, schema))
        conn.commit()

        total = 0
        batch: List[Dict[str, Any]] = []
        for feature in iter_features(paths):
            batch.append(feature)
            if len(batch) >= args.batch_size:
                total += insert_batch(conn, args.table, schema, batch)
                conn.commit()
                batch = []
                print(f"Inserted {total} rows into {args.table}")
        total += insert_batch(conn, args.table, schema, batch)
        conn.commit()
        print(f"Done. Inserted {total} rows from {len(paths)} files into {args.table}.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
