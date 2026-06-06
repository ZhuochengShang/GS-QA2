# QARV

QARV is the release bundle for the GS-QA2 vector/raster question-answering
experiments. It includes benchmark templates, compact result artifacts,
baseline runners, raster evaluation code, CHESS-PostGIS adaptation scripts,
and GISCopilot integration code.

## Contents

- `GS-QA/`: GS-QA baseline runners, evaluation scripts, QA2 templates, and
  compact experiment outputs.
- `GS-QA/benchmark/qa2/`: raster-only, raster-vector, and extended
  raster-vector question templates.
- `GS-QA/baselines/evaluation/`: evaluation outputs used to summarize vector
  and raster accuracy.
- `GS-QA/baselines/exp_tables/`: aggregate tables for reported metrics.
- `GS-QA/baselines/text2sql_gemini/`: collected Gemini Text2SQL vector
  outputs used by the evaluation scripts.
- `CHESS-PostGIS/`: CHESS adaptation for PostGIS, including raster-aware
  prompts, configs, and collection/evaluation utilities.
- `GISCopilot/`: cleaned GISCopilot source and GS-QA evaluation helpers.

## Large Data Not Included

Large data products are intentionally excluded from this repository:

- raw OSM extracts and DEM raster files
- PostGIS database dumps
- Chroma vector stores and embedding indexes
- full model caches and execution logs
- address caches and logs

Release large data artifacts separately, for example with Git LFS, Zenodo, or
an institutional data repository. The scripts expect raw data, database, and
index paths to be supplied through command-line arguments or environment
variables.

## Required Environment

The experiments require Python, PostGIS, and API credentials for the selected
LLM provider. Do not commit real API keys. Set them at runtime, for example:

```bash
export GEMINI_API_KEY="..."
export OPENAI_API_KEY="..."
```

For Qwen served through an OpenAI-compatible local endpoint, set:

```bash
export OPENAI_BASE_URL="http://HOST:PORT/v1"
export OPENAI_API_KEY="EMPTY"
```

## Cluster Experiment Commands

The commands below summarize the cluster runs used for the GS-QA vector/raster
experiments. Paths are examples from the UCR cluster setup and should be
adjusted if the benchmark, database, or raster files are stored elsewhere.

### GISCopilot on GS-QA

Use the `spatialagent` conda environment and run from the cleaned GISCopilot
evaluation directory:

```bash
cd ~/SpatialAnalysisAgent-master/SpatialAnalysisAgent
source ~/anaconda3/etc/profile.d/conda.sh
conda activate spatialagent

export QT_QPA_PLATFORM=offscreen
export XDG_RUNTIME_DIR=/tmp/runtime-zshan011
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

export PROJ_LIB=/home/zshan011/anaconda3/envs/spatialagent/share/proj
export GDAL_DATA=/home/zshan011/anaconda3/envs/spatialagent/share/gdal
export SPATIALAGENT_DATA_PATH=/home/zshan011/OsmData
export SPATIALAGENT_TASK_TIMEOUT=360
```

Run the full GS-QA T1-T28 benchmark with Gemini 2.5 Flash:

```bash
tmux new -s gis_t1_t28

python auto_eval_GSQA.py \
  --tasks-json workspace/tasks/GSQA_T1_T28_tasks.jsonl \
  --all \
  --model gemini-2.5-flash \
  --run-only \
  2>&1 | tee logs/gis_copilot_gemini25flash_T1_T28_360s_3tries_$(date -u +%Y%m%dT%H%M%SZ).log
```

For only T10-T28:

```bash
python auto_eval_GSQA.py \
  --tasks-json workspace/tasks/GSQA_T10_T28_tasks.jsonl \
  --all \
  --model gemini-2.5-flash \
  --run-only \
  2>&1 | tee logs/gis_copilot_gemini25flash_T10_T28_360s_3tries_$(date -u +%Y%m%dT%H%M%SZ).log
```

### GS-QA Text2SQL Baseline

Use the `postgis` conda environment and run from the GS-QA repository root:

```bash
cd ~/GS-QA-experiment/GS-QA
source ~/anaconda3/etc/profile.d/conda.sh
conda activate postgis

for i in $(seq 1 28); do
  task="T${i}"
  echo "===== Running ${task} ====="

  /home/zshan011/anaconda3/envs/postgis/bin/python baselines/baselines.py \
    --model gemini \
    --baseline text2sql \
    --questions-source benchmark \
    --benchmark-root benchmark \
    --benchmark-tasks "${task}" \
    --task-timeout-seconds 360 \
    --clear-cache sql_exec,sql_answer,sql_json_parse \
    --pg-host localhost \
    --pg-database gsqa \
    --pg-user shahd \
    --pg-password '' \
    --pg-port 5432 \
  || echo "FAILED ${task}"
done
```

### GS-QA RAG Baseline

Run from the baseline directory:

```bash
cd ~/GS-QA-experiment/GS-QA/baselines
conda activate postgis

for i in $(seq 1 28); do
  task="T${i}"
  out="../gemini_rag/${task}"

  /home/zshan011/anaconda3/envs/postgis/bin/python baselines.py \
    --model gemini \
    --parser-model gemini \
    --baseline rag \
    --embeddings minilm \
    --benchmark-root ~/GS-QA-experiment/GS-QA/benchmark \
    --benchmark-tasks "${task}" \
    --output-subdir "${out}" \
    --task-timeout-seconds 360 \
  || echo "FAILED: ${task}"
done
```

Check which RAG tasks completed:

```bash
for i in $(seq 1 28); do
  task="T${i}"
  dir="cache/gemini_rag/${task}"

  if [[ -f "${dir}/gemini_rag_text_eval.csv" && -f "${dir}/gemini_rag_parsed_eval.csv" ]]; then
    echo "SUCCESS ${task}"
  else
    echo "MISSING ${task}"
  fi
done
```

### Raster RAG Baseline

Build the DEM patch list for questions:

```bash
python find_question_dem_patches.py \
  --input-dir raster_only \
  --input-dir raster_vector \
  --input-dir extended \
  --dem-dir /local_data/scratch/zshan011/raster/dem/raster \
  --dem-glob '*_dem.tif' \
  --patch-size 256 \
  --dem-list-output /local_data/scratch/zshan011/raster/needed_dem_tiles.txt \
  --patches-output /local_data/scratch/zshan011/raster/needed_dem_patches.jsonl
```

Embed DEM patches:

```bash
python build_dem_patch_embeddings.py \
  --dem '/local_data/scratch/zshan011/raster/needed_dem_links/*_dem.tif' \
  --slope-dir /local_data/scratch/zshan011/raster/slope \
  --persist-directory /local_data/scratch/zshan011/raster/vectorstore_dem_patches_question_tiles \
  --collection-name dem_patches \
  --embedding-provider sentence-transformers \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --patch-size 256 \
  --batch-size 128 \
  --skip-missing-slope
```

Embed vector entities:

```bash
python build_vector_entity_embeddings.py \
  --input-dir raster_only \
  --input-dir raster_vector \
  --input-dir extended \
  --persist-directory /local_data/scratch/zshan011/raster/vectorstore_geo_entities \
  --collection-name geo_entities \
  --embedding-provider sentence-transformers \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --batch-size 128
```

Run raster RAG:

```bash
python run_raster_rag.py \
  --input-dir raster_only \
  --entity-persist-directory /local_data/scratch/zshan011/raster/vectorstore_geo_entities \
  --entity-collection-name geo_entities \
  --dem-persist-directory /local_data/scratch/zshan011/raster/vectorstore_dem_patches_question_tiles \
  --dem-collection-name dem_patches \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --top-k-entities 5 \
  --top-k-dem 10 \
  --spatial-prefilter \
  --llm-provider gemini \
  --model gemini-2.5-flash \
  --output rag/output/rag_raster_only_gemini_v3.jsonl \
  --resume
```

Do not commit API keys or generated artifacts such as logs, Chroma stores,
database dumps, or model outputs.
