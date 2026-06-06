#!/usr/bin/env bash
set -u
export GSQA_RASTER_RAG_ROOT=/home/zshan011/GS-QA

PY=/home/zshan011/anaconda3/envs/postgis/bin/python
BASE=/home/zshan011/GS-QA-experiment/GS-QA
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
    --dem-persist-directory /home/zshan011/GS-QA/dem_patches_gdal_chroma \
    --dem-collection-name dem_patches \
    --entity-persist-directory /home/zshan011/GS-QA/vector_entities_chroma \
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
