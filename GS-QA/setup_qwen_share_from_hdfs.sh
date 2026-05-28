#!/usr/bin/env bash
set -euo pipefail

HDFS_DIR="${HDFS_DIR:-/user/zshan011/share_qwen}"
WORK_DIR="${WORK_DIR:-$HOME/qwen_gsqa_share}"

mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

echo "[setup] downloading Qwen GS-QA/CHESS bundle from HDFS: $HDFS_DIR"
hdfs dfs -get -f "$HDFS_DIR/CHESS_code_qwen.tar.gz" .
hdfs dfs -get -f "$HDFS_DIR/GS_QA_experiment_qwen.tar.gz" .
hdfs dfs -get -f "$HDFS_DIR/gsqa_vector_raster_chroma_embeddings.tar.gz" .
hdfs dfs -get -f "$HDFS_DIR/gsqa_vector_raster_chroma_embeddings.tar.gz.sha256" .

echo "[setup] extracting code archives"
tar -xzf CHESS_code_qwen.tar.gz
tar -xzf GS_QA_experiment_qwen.tar.gz

echo "[setup] verifying embedding archive"
sha256sum -c gsqa_vector_raster_chroma_embeddings.tar.gz.sha256

echo "[setup] extracting embeddings"
tar -xzf gsqa_vector_raster_chroma_embeddings.tar.gz
mkdir -p GS-QA-experiment/GS-QA/shared_embeddings
rm -rf GS-QA-experiment/GS-QA/shared_embeddings/vector_entities_chroma
rm -rf GS-QA-experiment/GS-QA/shared_embeddings/dem_patches_gdal_chroma
mv vector_entities_chroma GS-QA-experiment/GS-QA/shared_embeddings/
mv dem_patches_gdal_chroma GS-QA-experiment/GS-QA/shared_embeddings/

chmod +x GS-QA-experiment/GS-QA/run_qwen_all_experiments.sh
chmod +x GS-QA-experiment/GS-QA/baselines/run_qwen_raster_rag_baselines.sh

cat <<EOF
[setup] done.

Code and embeddings are ready at:
  $WORK_DIR/GS-QA-experiment/GS-QA
  $WORK_DIR/CHESS/CHESS

Next command:
  cd "$WORK_DIR/GS-QA-experiment/GS-QA"
  MODEL_PATH=/path/to/Qwen3-32B-AWQ ./run_qwen_all_experiments.sh

For a smoke test only:
  cd "$WORK_DIR/GS-QA-experiment/GS-QA"
  MODEL_PATH=/path/to/Qwen3-32B-AWQ TASKS=1 QA2_SPLITS=raster_only ./run_qwen_all_experiments.sh
EOF
