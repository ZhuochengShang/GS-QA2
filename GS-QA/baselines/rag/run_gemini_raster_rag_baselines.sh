#!/usr/bin/env bash
set -u
BASE="${BASE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export GSQA_RASTER_RAG_ROOT="${GSQA_RASTER_RAG_ROOT:-$BASE}"

PY="${PY:-python3}"
ENTITY_STORE="${ENTITY_STORE:-$BASE/shared_embeddings/vector_entities_chroma}"
DEM_STORE="${DEM_STORE:-$BASE/shared_embeddings/dem_patches_gdal_chroma}"
cd "${BASE}/baselines"

run_one() {
  local subset="$1"
  echo "===== raster_rag ${subset} $(date) ====="
  "${PY}" baselines.py \
    --model gemini \
    --baseline raster_rag \
    --embeddings minilm \
    --benchmark-root "${BASE}/benchmark/qa2" \
    --benchmark-tasks "${subset}" \
    --output-subdir "../gemini_rag_raster/${subset}" \
    --dem-persist-directory "$DEM_STORE" \
    --dem-collection-name dem_patches \
    --entity-persist-directory "$ENTITY_STORE" \
    --entity-collection-name geo_entities \
    --top-k-entities 5 \
    --top-k-dem 5 \
    --spatial-candidate-limit 200 \
    --task-timeout-seconds 360
  echo "===== done ${subset} $(date) ====="
}

run_one raster_only
run_one raster_vector
run_one extended

echo "===== all raster_rag done $(date) ====="
