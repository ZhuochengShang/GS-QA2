#!/usr/bin/env bash
set -euo pipefail

GS_QA_ROOT="${GS_QA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO_ROOT="${REPO_ROOT:-$(cd "$GS_QA_ROOT/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the directory containing osm_extract/}"

PGHOST="${PGHOST:-localhost}"
PGPORT="${PGPORT:-5432}"
PGDATABASE="${PGDATABASE:-gsqa}"
PGUSER="${PGUSER:-postgres}"
PGPASSWORD="${PGPASSWORD:-}"
export PGPASSWORD

LOADER="$REPO_ROOT/CHESS-PostGIS/scripts/load_gsqa_osm_geojson_to_postgis.py"

run_loader() {
  local table="$1"
  local glob_path="$2"
  local schema="$3"
  python "$LOADER" \
    --geojson-glob "$glob_path" \
    --table "$table" \
    --schema-json "$schema" \
    --db-host "$PGHOST" \
    --db-port "$PGPORT" \
    --db-name "$PGDATABASE" \
    --db-user "$PGUSER" \
    --db-password "$PGPASSWORD"
}

run_loader pois    "$DATA_ROOT/osm_extract/pois/*.geojson"         "$GS_QA_ROOT/generator/poi_schema.json"
run_loader roads   "$DATA_ROOT/osm_extract/roads/*.geojson"        "$GS_QA_ROOT/generator/roads_schema.json"
run_loader parks   "$DATA_ROOT/osm_extract/parks/*.geojson"        "$GS_QA_ROOT/generator/parks_schema.json"
run_loader lakes   "$DATA_ROOT/osm_extract/lakes/*.geojson"        "$GS_QA_ROOT/generator/lakes_schema.json"
run_loader regions "$DATA_ROOT/osm_extract/postal_codes/*.geojson" "$GS_QA_ROOT/generator/region_schema.json"

psql "host=$PGHOST port=$PGPORT dbname=$PGDATABASE user=$PGUSER" <<'SQL'
ANALYZE pois;
ANALYZE roads;
ANALYZE parks;
ANALYZE lakes;
ANALYZE regions;
SQL
