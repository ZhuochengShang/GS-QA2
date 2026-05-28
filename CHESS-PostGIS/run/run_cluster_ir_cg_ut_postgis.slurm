#!/bin/bash
#SBATCH --job-name=chess_gsqa_cg_ut
#SBATCH --array=1-28
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/chess_gsqa_T%a_%j.out
#SBATCH --error=logs/slurm/chess_gsqa_T%a_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
cd "$REPO_DIR"

mkdir -p logs/slurm

if [ -f ".env" ]; then
  set -a
  source .env
  set +a
fi

ENV_NAME="${ENV_NAME:-chess-postgis}"

# Uncomment and edit module loading if your cluster requires it before conda.
# module load miniconda3

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "$ENV_NAME"
else
  echo "conda is not available on PATH. Load your cluster conda module before running this job." >&2
  exit 1
fi

SPLIT_ID="${SLURM_ARRAY_TASK_ID:-1}"
DATA_MODE="${DATA_MODE:-dev}"
DATA_PATH="${DATA_PATH:-data/dev/gsqa_T${SPLIT_ID}_postgis.json}"
CONFIG_PATH="${CONFIG_PATH:-run/configs/CHESS_IR_CG_UT.yaml}"
NUM_WORKERS="${NUM_WORKERS:-1}"
LOG_LEVEL="${LOG_LEVEL:-info}"

PG_DATABASE="${PG_DATABASE:-${CHESS_PG_DATABASE:-gsqa}}"
PG_HOST="${PG_HOST:-${CHESS_PG_HOST:-localhost}}"
PG_USER="${PG_USER:-${CHESS_PG_USER:-clockorangezoe}}"
PG_PASSWORD="${PG_PASSWORD:-${CHESS_PG_PASSWORD:-}}"
PG_PORT="${PG_PORT:-${CHESS_PG_PORT:-5432}}"

echo "Running GS-QA T${SPLIT_ID}"
echo "DATA_PATH=${DATA_PATH}"
echo "CONFIG_PATH=${CONFIG_PATH}"
echo "PG_HOST=${PG_HOST}"
echo "PG_DATABASE=${PG_DATABASE}"
echo "PG_USER=${PG_USER}"
echo "PG_PORT=${PG_PORT}"

python3 -u src/main.py \
  --data_mode "$DATA_MODE" \
  --data_path "$DATA_PATH" \
  --config "$CONFIG_PATH" \
  --num_workers "$NUM_WORKERS" \
  --log_level "$LOG_LEVEL" \
  --pick_final_sql true \
  --sql_dialect postgis \
  --pg_database "$PG_DATABASE" \
  --pg_host "$PG_HOST" \
  --pg_user "$PG_USER" \
  --pg_password "$PG_PASSWORD" \
  --pg_port "$PG_PORT"
