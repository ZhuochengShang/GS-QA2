#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_DIR"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  PYTHON_BIN=python3
fi

BENCHMARK_ROOT="${BENCHMARK_ROOT:-$HOME/GS-QA-experiment/GS-QA/benchmark/qa2}"
RASTER_ONLY_INPUT_DIR="${RASTER_ONLY_INPUT_DIR:-$BENCHMARK_ROOT/raster_only}"
RASTER_VECTOR_INPUT_DIR="${RASTER_VECTOR_INPUT_DIR:-$BENCHMARK_ROOT/raster_vector}"
EXTENDED_INPUT_DIR="${EXTENDED_INPUT_DIR:-$BENCHMARK_ROOT/extended}"

OUT_DIR="${OUT_DIR:-data/dev}"
DB_ID="${DB_ID:-osm}"
mkdir -p "$OUT_DIR"

echo "Writing raster-only dataset from: $RASTER_ONLY_INPUT_DIR"
"$PYTHON_BIN" scripts/convert_raster_gsqa_to_chess_postgis.py \
  --input-dir "$RASTER_ONLY_INPUT_DIR" \
  --output "$OUT_DIR/gsqa_raster_only_postgis.json" \
  --db-id "$DB_ID"

echo "Writing raster-vector dataset from: $RASTER_VECTOR_INPUT_DIR"
"$PYTHON_BIN" scripts/convert_raster_gsqa_to_chess_postgis.py \
  --input-dir "$RASTER_VECTOR_INPUT_DIR" \
  --output "$OUT_DIR/gsqa_raster_vector_postgis.json" \
  --db-id "$DB_ID"

echo "Writing extended dataset from: $EXTENDED_INPUT_DIR"
"$PYTHON_BIN" scripts/convert_raster_gsqa_to_chess_postgis.py \
  --input-dir "$EXTENDED_INPUT_DIR" \
  --output "$OUT_DIR/gsqa_extended_postgis.json" \
  --db-id "$DB_ID"
