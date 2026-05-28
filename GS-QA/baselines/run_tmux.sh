#!/usr/bin/env bash
# run_tmux.sh — launch one tmux window per query type
# Usage: bash run_tmux.sh [--workers N] [--samples N]
#
# Creates a tmux session called "gsqa" with three windows:
#   window 0: raster_only
#   window 1: raster_vector
#   window 2: extended

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults — edit these or override via env
# ---------------------------------------------------------------------------
SESSION="gsqa"
SCRIPT="baselines/run_raster_text2sql.py"
BENCHMARK_DIR="${BENCHMARK_DIR:-benchmark/qa2}"
CACHE_DIR="baselines/cache"
WORKERS="${WORKERS:-6}"
SAMPLES="${SAMPLES:-0}"          # 0 = all questions
DB_NAME="${DB_NAME:-gsqa}"
DB_USER="${DB_USER:-zshan011}"
DB_HOST="${DB_HOST:-localhost}"
LOG_DIR="baselines/logs"

# ---------------------------------------------------------------------------
# Parse optional flags
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case $1 in
    --workers)  WORKERS="$2";  shift 2 ;;
    --samples)  SAMPLES="$2";  shift 2 ;;
    --db-name)  DB_NAME="$2";  shift 2 ;;
    --db-user)  DB_USER="$2";  shift 2 ;;
    --db-host)  DB_HOST="$2";  shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
if [[ -z "${GEMINI_API_KEY:-}" ]]; then
  echo "ERROR: GEMINI_API_KEY is not set."
  echo "  export GEMINI_API_KEY=your_key_here"
  exit 1
fi

if ! command -v tmux &>/dev/null; then
  echo "ERROR: tmux not found. Install with: sudo apt install tmux"
  exit 1
fi

if ! command -v python3 &>/dev/null; then
  echo "ERROR: python3 not found."
  exit 1
fi

mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Base command shared across all windows
# ---------------------------------------------------------------------------
BASE_CMD="python3 $SCRIPT \
  --benchmark-dir \"$BENCHMARK_DIR\" \
  --cache-dir \"$CACHE_DIR\" \
  --workers $WORKERS \
  --samples-per-file $SAMPLES \
  --db-name $DB_NAME \
  --db-user $DB_USER \
  --db-host $DB_HOST \
  --generation-timeout 360 \
  --execution-timeout 180"

# ---------------------------------------------------------------------------
# Kill existing session if it exists, then create fresh
# ---------------------------------------------------------------------------
tmux kill-session -t "$SESSION" 2>/dev/null || true

echo "Starting tmux session: $SESSION"
echo "  workers per query type : $WORKERS"
echo "  samples per file       : ${SAMPLES:-all}"
echo "  cache dir              : $CACHE_DIR"
echo ""

# window 0: raster_only
tmux new-session  -d -s "$SESSION" -n "raster_only"
tmux send-keys -t "$SESSION:raster_only" \
  "export GEMINI_API_KEY=$GEMINI_API_KEY && \
   $BASE_CMD --query-types raster_only 2>&1 | tee $LOG_DIR/raster_only.log; \
   echo '=== raster_only DONE ===' " Enter

# window 1: raster_vector
tmux new-window -t "$SESSION" -n "raster_vector"
tmux send-keys -t "$SESSION:raster_vector" \
  "export GEMINI_API_KEY=$GEMINI_API_KEY && \
   $BASE_CMD --query-types raster_vector 2>&1 | tee $LOG_DIR/raster_vector.log; \
   echo '=== raster_vector DONE ===' " Enter

# window 2: extended
tmux new-window -t "$SESSION" -n "extended"
tmux send-keys -t "$SESSION:extended" \
  "export GEMINI_API_KEY=$GEMINI_API_KEY && \
   $BASE_CMD --query-types extended 2>&1 | tee $LOG_DIR/extended.log; \
   echo '=== extended DONE ===' " Enter

# go back to first window
tmux select-window -t "$SESSION:raster_only"

echo "tmux session '$SESSION' started with 3 windows."
echo ""
echo "Attach:          tmux attach -t $SESSION"
echo "Switch windows:  Ctrl-b then 0 / 1 / 2"
echo "Detach (keep running): Ctrl-b then d"
echo ""
echo "Tail logs without attaching:"
echo "  tail -f $LOG_DIR/raster_only.log"
echo "  tail -f $LOG_DIR/raster_vector.log"
echo "  tail -f $LOG_DIR/extended.log"

# Optionally attach immediately
if [[ "${ATTACH:-1}" == "1" ]]; then
  tmux attach -t "$SESSION"
fi