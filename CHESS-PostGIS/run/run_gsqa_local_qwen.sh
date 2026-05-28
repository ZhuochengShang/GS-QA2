#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_DIR"

SPLIT_ID="${SPLIT_ID:-1}"
DATA_MODE="${DATA_MODE:-dev}"
DATA_PATH="${DATA_PATH:-data/dev/gsqa_T${SPLIT_ID}_postgis.json}"
CONFIG_PATH="${CONFIG_PATH:-run/configs/CHESS_IR_CG_UT_QWEN_LOCAL.yaml}"
NUM_WORKERS="${NUM_WORKERS:-1}"
LOG_LEVEL="${LOG_LEVEL:-info}"

export CHESS_SQL_DIALECT="${CHESS_SQL_DIALECT:-postgis}"
export DB_ROOT_PATH="${DB_ROOT_PATH:-$REPO_DIR/data/dev}"
export INDEX_SERVER_HOST="${INDEX_SERVER_HOST:-localhost}"
export INDEX_SERVER_PORT="${INDEX_SERVER_PORT:-8000}"
export CHESS_FIXED_SCHEMA_FILE="${CHESS_FIXED_SCHEMA_FILE:-$REPO_DIR/templates/gsqa_postgis_schema.txt}"
export CHESS_DB_TABLE_ALLOWLIST="${CHESS_DB_TABLE_ALLOWLIST:-pois,lakes,parks,roads,regions}"

export LOCAL_LLM_MODEL="${LOCAL_LLM_MODEL:-Qwen/Qwen3-32B-AWQ}"
export LOCAL_LLM_BASE_URL="${LOCAL_LLM_BASE_URL:-http://localhost:8000/v1}"
export LOCAL_LLM_API_KEY="${LOCAL_LLM_API_KEY:-EMPTY}"
export LOCAL_LLM_MAX_TOKENS="${LOCAL_LLM_MAX_TOKENS:-4096}"

PG_HOST="${PG_HOST:-localhost}"
PG_DATABASE="${PG_DATABASE:-gsqa}"
PG_USER="${PG_USER:-zshan011}"
PG_PASSWORD="${PG_PASSWORD:-}"
PG_PORT="${PG_PORT:-5432}"

echo "Running CHESS GS-QA T${SPLIT_ID} with local OpenAI-compatible model"
echo "LOCAL_LLM_MODEL=${LOCAL_LLM_MODEL}"
echo "LOCAL_LLM_BASE_URL=${LOCAL_LLM_BASE_URL}"
echo "DATA_PATH=${DATA_PATH}"
echo "CONFIG_PATH=${CONFIG_PATH}"

python3 -u src/main.py \
  --data_mode "$DATA_MODE" \
  --data_path "$DATA_PATH" \
  --config "$CONFIG_PATH" \
  --num_workers "$NUM_WORKERS" \
  --log_level "$LOG_LEVEL" \
  --pick_final_sql true \
  --sql_dialect postgis \
  --pg_host "$PG_HOST" \
  --pg_database "$PG_DATABASE" \
  --pg_user "$PG_USER" \
  --pg_password "$PG_PASSWORD" \
  --pg_port "$PG_PORT"
