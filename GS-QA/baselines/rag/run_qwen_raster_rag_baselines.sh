#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH="${MODEL_PATH:-/path/to/Qwen3-32B-AWQ}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:8000/v1}"
OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
SAMPLES_PER_FILE="${SAMPLES_PER_FILE:-100000}"

ENTITY_STORE="${ENTITY_STORE:-$BASE/shared_embeddings/vector_entities_chroma}"
DEM_STORE="${DEM_STORE:-$BASE/shared_embeddings/dem_patches_gdal_chroma}"
OUT_ROOT="${OUT_ROOT:-$BASE/baselines/cache/qwen_rag_raster}"

cd "$BASE"

run_one() {
  local subset="$1"
  mkdir -p "$OUT_ROOT/$subset"
  echo "===== qwen raster_rag ${subset} $(date) ====="
  OPENAI_BASE_URL="$OPENAI_BASE_URL" OPENAI_API_KEY="$OPENAI_API_KEY" \
    "$PYTHON_BIN" baselines/rag/run_raster_rag.py \
      --input-dir "$BASE/benchmark/qa2/$subset" \
      --output "$OUT_ROOT/$subset/raster_rag.jsonl" \
      --dem-persist-directory "$DEM_STORE" \
      --dem-collection-name dem_patches \
      --entity-persist-directory "$ENTITY_STORE" \
      --entity-collection-name geo_entities \
      --embedding-provider sentence-transformers \
      --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
      --samples-per-file "$SAMPLES_PER_FILE" \
      --top-k-entities 5 \
      --top-k-dem 5 \
      --spatial-prefilter \
      --spatial-candidate-limit 200 \
      --llm-provider openai \
      --model "$MODEL_PATH" \
      --resume
  echo "===== done ${subset} $(date) ====="
}

if [[ $# -gt 0 ]]; then
  for subset in "$@"; do
    run_one "$subset"
  done
else
  run_one raster_only
  run_one raster_vector
  run_one extended
fi

echo "===== all qwen raster_rag done $(date) ====="
