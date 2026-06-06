# QARV

QARV is a release repository for GS-QA2, a geospatial question-answering
benchmark over vector and raster data. The main contribution is the benchmark
construction pipeline: natural-language templates, executable SQL templates,
ground-truth answer generation, and evaluation artifacts for vector-only,
raster-only, and raster-vector questions.

## Repository Layout

- `GS-QA/benchmark/qa2/`: GS-QA2 raster-only, raster-vector, and extended
  raster-vector question files. Each JSONL record contains the question text,
  answer type, SQL/ground-truth fields, and template metadata.
- `GS-QA/generator/`: vector benchmark generation code and template logic.
- `GS-QA/baselines/`: Text2SQL, RAG, raster Text2SQL, raster RAG, evaluation,
  and result aggregation scripts.
- `GS-QA/baselines/evaluation/`: compact evaluation outputs used to summarize
  accuracy and failure modes.
- `GS-QA/baselines/exp_tables/`: aggregate result tables.
- `GS-QA/baselines/text2sql_gemini/`: collected Text2SQL result artifacts for
  vector location templates.
- `CHESS-PostGIS/`: CHESS adaptation for PostGIS, including GS-QA vector and
  raster run scripts.
- `GISCopilot/`: GIS Copilot source and GS-QA evaluation helpers.

## Data Not Included

Large data products are not committed:

- raw OSM extracts and DEM raster files
- PostGIS database dumps
- vector-store and embedding indexes
- full model caches, runtime logs, and temporary execution outputs
- address-geocoding caches

The scripts expect these paths to be supplied through command-line arguments or
environment variables. Use placeholders such as `$GS_QA_ROOT`, `$DATA_ROOT`,
`$DEM_ROOT`, `$VECTORSTORE_ROOT`, and `$SCRATCH_ROOT` in local run scripts.

## Environment

Install Python dependencies from the baseline/generator folders as needed:

```bash
cd GS-QA
python -m pip install -r generator/requirements.txt
python -m pip install -r baselines/requirements.txt
```

The database-backed baselines require PostgreSQL with PostGIS enabled:

```sql
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_raster;
```

Set model credentials at runtime. Do not commit real keys:

```bash
export GEMINI_API_KEY="..."
export OPENAI_API_KEY="..."
```

For a locally served Qwen model using an API-compatible endpoint:

```bash
export LOCAL_LLM_BASE_URL="http://localhost:8000/v1"
export LOCAL_LLM_API_KEY="EMPTY"
export LOCAL_LLM_MODEL="Qwen/Qwen3-32B-AWQ"
```

## Building the Benchmark

### Vector Data Ingestion

Prepare OSM-derived GeoJSON files with the expected layer structure:

```text
$DATA_ROOT/osm_extract/
  lakes/
  parks/
  pois/
  postal_codes/
  roads/
```

Load the vector layers into PostGIS using the processors in `GS-QA/generator/`.
Each processor maps a GeoJSON layer to a benchmark table using the matching
schema file, for example `poi_schema.json` for POIs.

```bash
cd GS-QA/generator
python pois_processor.py
python roads_processor.py
python parks_processor.py
python lakes_processor.py
python regions_processor.py
```

Configure the database connection in the processor scripts before running them.
The benchmark uses OSM feature tables for points, roads, parks, water bodies,
and regions, plus selected non-spatial attributes such as category, name, and
Wikipedia-derived metadata.

### Raster Data Ingestion

Load DEM rasters into PostGIS as tiled raster tables. A typical ingestion flow is:

```bash
raster2pgsql -s 4326 -I -C -M -t 256x256 "$DEM_ROOT"/*.tif public.dem_tiles \
  | psql "$POSTGIS_DSN"
```

The raster tables should preserve tile extent, CRS, pixel resolution, and band
statistics. Build GiST indexes on raster convex hulls and geometry columns so
local lookup, raster-vector overlay, and spatial prefiltering can use database
indexes.

### Template and Ground-Truth Generation

Vector templates are generated from `GS-QA/generator/templates/` and the
generator code in `GS-QA/generator/`. Each template defines:

- the natural-language pattern
- slots for entities, distances, directions, regions, or categories
- an executable SQL template
- the answer type and verifier

Raster and raster-vector templates are stored in `GS-QA/benchmark/qa2/`. These
files include local DEM lookup, focal terrain attributes, zonal statistics,
global terrain search, and mixed vector-raster predicates. The SQL fields serve
as the ground-truth programs used to compute reference answers.

## Running Baselines

Set common database variables first:

```bash
export PG_HOST="localhost"
export PG_PORT="5432"
export PG_DATABASE="gsqa"
export PG_USER="postgres"
export PG_PASSWORD=""
```

### Text2SQL

Run vector Text2SQL over T1-T28:

```bash
cd GS-QA
for i in $(seq 1 28); do
  python baselines/baselines.py \
    --model gemini \
    --baseline text2sql \
    --questions-source benchmark \
    --benchmark-root benchmark \
    --benchmark-tasks "T${i}" \
    --task-timeout-seconds 360 \
    --pg-host "$PG_HOST" \
    --pg-port "$PG_PORT" \
    --pg-database "$PG_DATABASE" \
    --pg-user "$PG_USER" \
    --pg-password "$PG_PASSWORD"
done
```

Run raster Text2SQL:

```bash
cd GS-QA/baselines
python run_raster_text2sql.py \
  --benchmark-dir ../benchmark/qa2 \
  --query-types raster_only raster_vector extended \
  --provider gemini \
  --model gemini-2.5-flash \
  --db-host "$PG_HOST" \
  --db-port "$PG_PORT" \
  --db-name "$PG_DATABASE" \
  --db-user "$PG_USER" \
  --db-password "$PG_PASSWORD" \
  --cache-dir cache
```

### RAG

Run vector RAG over T1-T28:

```bash
cd GS-QA/baselines
for i in $(seq 1 28); do
  python baselines.py \
    --model gemini \
    --parser-model gemini \
    --baseline rag \
    --embeddings minilm \
    --benchmark-root ../benchmark \
    --benchmark-tasks "T${i}" \
    --output-subdir "cache/gemini_rag/T${i}" \
    --task-timeout-seconds 360
done
```

For raster RAG, provide prebuilt vector-entity and DEM-patch stores. If the
question-relevant DEM patch list is available as JSONL, build the DEM patch
store with:

```bash
cd GS-QA/baselines
python build_question_dem_patch_embeddings.py \
  --patches "$SCRATCH_ROOT/needed_dem_patches.jsonl" \
  --persist-directory "$VECTORSTORE_ROOT/dem_patches" \
  --collection-name dem_patches \
  --embedding-provider sentence-transformers \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2
```

Then run retrieval and answer generation:

```bash
python run_raster_rag.py \
  --input-dir ../benchmark/qa2/raster_only \
  --input-dir ../benchmark/qa2/raster_vector \
  --input-dir ../benchmark/qa2/extended \
  --entity-persist-directory "$VECTORSTORE_ROOT/geo_entities" \
  --dem-persist-directory "$VECTORSTORE_ROOT/dem_patches" \
  --llm-provider gemini \
  --model gemini-2.5-flash \
  --output cache/gemini_rag_raster/raster_rag.jsonl \
  --resume
```

### CHESS-PostGIS

CHESS-PostGIS adapts CHESS from SQLite-style text-to-SQL to PostGIS. It uses a
static schema, generates PostGIS SQL, revises failed candidates, and selects a
final query using execution results and generated unit tests.

Run vector templates:

```bash
cd CHESS-PostGIS
export DATA_MODE="dev"
export PG_HOST="$PG_HOST"
export PG_PORT="$PG_PORT"
export PG_DATABASE="$PG_DATABASE"
export PG_USER="$PG_USER"
export PG_PASSWORD="$PG_PASSWORD"

for i in $(seq 1 28); do
  SPLIT_ID="$i" bash run/run_gsqa_local_qwen.sh
done
```

Run raster templates:

```bash
cd CHESS-PostGIS
BENCHMARK_ROOT="../GS-QA/benchmark/qa2" bash run/prepare_raster_chess_datasets.sh
CONFIG_PATH=run/configs/CHESS_IR_CG_UT_QWEN_LOCAL_RASTER.yaml \
  bash run/run_chess_raster_only_postgis.sh
CONFIG_PATH=run/configs/CHESS_IR_CG_UT_QWEN_LOCAL_RASTER.yaml \
  bash run/run_chess_raster_vector_postgis.sh
CONFIG_PATH=run/configs/CHESS_IR_CG_UT_QWEN_LOCAL_RASTER.yaml \
  bash run/run_chess_extended_postgis.sh
```

### GIS Copilot

GIS Copilot is evaluated as a tool-using code-generation baseline. It consumes
task JSONL files and dataset paths, then runs a headless geospatial workflow.

```bash
cd GISCopilot/SpatialAnalysisAgent
export QT_QPA_PLATFORM="offscreen"
export XDG_RUNTIME_DIR="/tmp/runtime-$USER"
export SPATIALAGENT_DATA_PATH="$DATA_ROOT/osm_extract"
export SPATIALAGENT_TASK_TIMEOUT="360"

python auto_eval_GSQA.py \
  --tasks-json workspace/tasks/GSQA_T1_T28_tasks.jsonl \
  --all \
  --model gemini-2.5-flash \
  --run-only
```

Raster tasks can be run with `auto_eval_raster.py` after providing the DEM and
vector input paths expected by the task file.

## Evaluation

Vector evaluation is implemented in `GS-QA/baselines/evaluate.py`. Raster and
compound raster-vector evaluation are implemented in
`GS-QA/baselines/evaluate_raster.py`.

Useful aggregation scripts:

```bash
cd GS-QA
python build_sql_error_summary.py
python collect_text2sql_results.py
python rescore_vector_raw.py
```

Published compact outputs are under `GS-QA/baselines/evaluation/` and
`GS-QA/baselines/exp_tables/`.
