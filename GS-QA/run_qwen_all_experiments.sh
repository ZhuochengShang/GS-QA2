#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
SHARE_ROOT="${SHARE_ROOT:-$(cd "$BASE/../.." && pwd)}"
DEFAULT_CHESS_ROOT="$BASE/baselines/chess/runtime/CHESS/CHESS"
if [[ ! -d "$DEFAULT_CHESS_ROOT" && -d "$SHARE_ROOT/CHESS/CHESS" ]]; then
  DEFAULT_CHESS_ROOT="$SHARE_ROOT/CHESS/CHESS"
fi
CHESS_ROOT="${CHESS_ROOT:-$DEFAULT_CHESS_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:8000/v1}"
OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
TASKS="${TASKS:-$(seq 1 28)}"
QA2_SPLITS="${QA2_SPLITS:-raster_only raster_vector extended}"

PG_HOST="${PG_HOST:-localhost}"
PG_DATABASE="${PG_DATABASE:-gsqa}"
PG_USER="${PG_USER:-$USER}"
PG_PASSWORD="${PG_PASSWORD:-}"
PG_PORT="${PG_PORT:-5432}"

RUN_CHESS_VECTOR="${RUN_CHESS_VECTOR:-1}"
RUN_CHESS_QA2="${RUN_CHESS_QA2:-1}"
RUN_RAG_VECTOR="${RUN_RAG_VECTOR:-1}"
RUN_RAG_QA2="${RUN_RAG_QA2:-1}"
RUN_TEXT2SQL_QA2="${RUN_TEXT2SQL_QA2:-1}"

if [[ -z "$MODEL_PATH" ]]; then
  echo "ERROR: set MODEL_PATH=/path/to/Qwen3-32B-AWQ" >&2
  exit 2
fi

echo "[run] BASE=$BASE"
echo "[run] CHESS_ROOT=$CHESS_ROOT"
echo "[run] MODEL_PATH=$MODEL_PATH"
echo "[run] OPENAI_BASE_URL=$OPENAI_BASE_URL"
echo "[run] TASKS=$TASKS"
echo "[run] QA2_SPLITS=$QA2_SPLITS"

echo "[run] checking Qwen server"
curl -fsS "$OPENAI_BASE_URL/models" >/dev/null

if [[ "$RUN_CHESS_VECTOR" == "1" ]]; then
  echo "[run] CHESS vector GS-QA benchmark"
  cd "$CHESS_ROOT"
  for i in $TASKS; do
    LOCAL_LLM_MODEL="$MODEL_PATH" \
    LOCAL_LLM_BASE_URL="$OPENAI_BASE_URL" \
    LOCAL_LLM_API_KEY="$OPENAI_API_KEY" \
    LOCAL_LLM_MAX_TOKENS="${LOCAL_LLM_MAX_TOKENS:-4096}" \
    SPLIT_ID="$i" \
    PG_HOST="$PG_HOST" \
    PG_DATABASE="$PG_DATABASE" \
    PG_USER="$PG_USER" \
    PG_PASSWORD="$PG_PASSWORD" \
    PG_PORT="$PG_PORT" \
    bash run/run_gsqa_local_qwen.sh
  done
fi

if [[ "$RUN_CHESS_QA2" == "1" ]]; then
  echo "[run] CHESS qa2 PostGIS"
  cd "$CHESS_ROOT"
  BENCHMARK_ROOT="$BASE/benchmark/qa2" bash run/prepare_raster_chess_datasets.sh
  for split in $QA2_SPLITS; do
    case "$split" in
      raster_only) script="run/run_chess_raster_only_postgis.sh" ;;
      raster_vector) script="run/run_chess_raster_vector_postgis.sh" ;;
      extended) script="run/run_chess_extended_postgis.sh" ;;
      *) echo "Unknown QA2 split: $split" >&2; exit 2 ;;
    esac
    LOCAL_LLM_MODEL="$MODEL_PATH" \
    LOCAL_LLM_BASE_URL="$OPENAI_BASE_URL" \
    LOCAL_LLM_API_KEY="$OPENAI_API_KEY" \
    LOCAL_LLM_MAX_TOKENS="${LOCAL_LLM_MAX_TOKENS:-4096}" \
    CONFIG_PATH="run/configs/CHESS_IR_CG_UT_QWEN_LOCAL_RASTER.yaml" \
    PG_HOST="$PG_HOST" \
    PG_DATABASE="$PG_DATABASE" \
    PG_USER="$PG_USER" \
    PG_PASSWORD="$PG_PASSWORD" \
    PG_PORT="$PG_PORT" \
    bash "$script"
  done
fi

if [[ "$RUN_RAG_VECTOR" == "1" ]]; then
  echo "[run] vector RAG GS-QA benchmark"
  cd "$BASE"
  for i in $TASKS; do
    "$PYTHON_BIN" baselines/baselines.py \
      --model "$MODEL_PATH" \
      --openai-base-url "$OPENAI_BASE_URL" \
      --openai-api-key "$OPENAI_API_KEY" \
      --baseline rag \
      --embeddings minilm \
      --questions-source benchmark \
      --benchmark-root benchmark \
      --benchmark-tasks "T${i}" \
      --task-timeout-seconds 360
  done
fi

if [[ "$RUN_RAG_QA2" == "1" ]]; then
  echo "[run] raster RAG qa2"
  cd "$BASE"
  MODEL_PATH="$MODEL_PATH" \
  OPENAI_BASE_URL="$OPENAI_BASE_URL" \
  OPENAI_API_KEY="$OPENAI_API_KEY" \
  bash baselines/rag/run_qwen_raster_rag_baselines.sh $QA2_SPLITS
fi

if [[ "$RUN_TEXT2SQL_QA2" == "1" ]]; then
  echo "[run] raster Text2SQL qa2"
  cd "$BASE"
  OPENAI_API_KEY="$OPENAI_API_KEY" "$PYTHON_BIN" baselines/text2sql/run_raster_text2sql.py \
    --benchmark-dir "$BASE/benchmark/qa2" \
    --query-types $QA2_SPLITS \
    --provider openai \
    --model "$MODEL_PATH" \
    --openai-base-url "$OPENAI_BASE_URL" \
    --generation-timeout 360 \
    --execution-timeout 360 \
    --db-host "$PG_HOST" \
    --db-name "$PG_DATABASE" \
    --db-user "$PG_USER" \
    --db-password "$PG_PASSWORD" \
    --db-port "$PG_PORT"
fi

cat <<EOF
[run] done.

Main output roots:
  CHESS vector:     $CHESS_ROOT/results/dev/CHESS_IR_CG_UT_QWEN_LOCAL/
  CHESS qa2:        $CHESS_ROOT/results/dev/CHESS_IR_CG_UT_QWEN_LOCAL_RASTER/
  vector RAG:       $BASE/baselines/cache/$(basename "$MODEL_PATH")/
  raster RAG:       $BASE/baselines/cache/qwen_rag_raster/
  raster Text2SQL:  $BASE/baselines/cache/openai_raster_only, openai_raster_vector, openai_extended
EOF
