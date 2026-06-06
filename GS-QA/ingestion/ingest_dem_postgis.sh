#!/usr/bin/env bash
set -euo pipefail

DEM_ROOT="${DEM_ROOT:?Set DEM_ROOT to the directory containing DEM GeoTIFF files}"
DEM_GLOB="${DEM_GLOB:-*.tif}"
DEM_TABLE="${DEM_TABLE:-public.dem_us}"
POSTGIS_DSN="${POSTGIS_DSN:?Set POSTGIS_DSN, e.g. postgresql://postgres@localhost:5432/gsqa}"
SRID="${SRID:-4326}"
TILE_SIZE="${TILE_SIZE:-256x256}"

raster2pgsql -s "$SRID" -I -C -M -t "$TILE_SIZE" "$DEM_ROOT"/$DEM_GLOB "$DEM_TABLE" \
  | psql "$POSTGIS_DSN"

psql "$POSTGIS_DSN" <<SQL
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_raster;
ANALYZE ${DEM_TABLE};
SQL
