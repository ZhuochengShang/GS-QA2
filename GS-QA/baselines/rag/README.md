# RAG and Embedding Workflow

The vector RAG baseline and raster RAG baseline use different stores.

## Vector-Only RAG

Vector-only RAG is implemented in `GS-QA/baselines/baselines.py`. When running
with `--baseline rag`, the script builds a Chroma store automatically if it does
not already exist.

```bash
cd GS-QA/baselines
python baselines.py \
  --model gemini \
  --parser-model gemini \
  --baseline rag \
  --embeddings minilm \
  --benchmark-root ../benchmark \
  --benchmark-tasks T1 \
  --output-subdir cache/gemini_rag/T1
```

The resulting store is written under:

```text
GS-QA/osm_vectorstore_minilm/
```

The embedded corpus contains OSM objects extracted from benchmark question
entities and optional Wikipedia corpus records when `wikipedia_corpus.jsonl` is
available.

## Raster RAG

Raster RAG uses two Chroma stores:

- a vector-entity store for OSM-like entities and geometries
- a DEM-patch store for raster metadata and patch windows

In the Qwen experiment package, these stores are distributed as a shared
embedding archive:

```text
gsqa_vector_raster_chroma_embeddings.tar.gz
gsqa_vector_raster_chroma_embeddings.tar.gz.sha256
```

Use the setup script to download and place the stores automatically:

```bash
mkdir -p ~/qwen_gsqa_share
cd ~/qwen_gsqa_share

hdfs dfs -get -f /user/$USER/share_qwen/GS_QA_experiment_qwen.tar.gz .
tar -xzf GS_QA_experiment_qwen.tar.gz

cd GS-QA-experiment/GS-QA
./setup_qwen_share_from_hdfs.sh
```

The script downloads the code and embedding archives from HDFS, verifies the
embedding archive checksum, and extracts the stores to:

```text
GS-QA-experiment/GS-QA/shared_embeddings/vector_entities_chroma/
GS-QA-experiment/GS-QA/shared_embeddings/dem_patches_gdal_chroma/
```

The archive is the expected path for reproducing the Qwen experiments. The
commands below rebuild the stores only when the archive is unavailable or the
benchmark/data files have changed.

The Qwen experiment wrapper uses these shared stores directly:

```bash
cd GS-QA
MODEL_PATH=/path/to/Qwen3-32B-AWQ \
RUN_CHESS_VECTOR=0 \
RUN_CHESS_QA2=0 \
RUN_RAG_VECTOR=0 \
RUN_RAG_QA2=1 \
RUN_TEXT2SQL_QA2=0 \
./run_qwen_all_experiments.sh
```

Build the vector entity store from QA2 question entities:

```bash
cd GS-QA
python baselines/rag/build_vector_entity_embeddings.py \
  --input-dir benchmark/qa2/raster_only \
  --input-dir benchmark/qa2/raster_vector \
  --input-dir benchmark/qa2/extended \
  --persist-directory shared_embeddings/vector_entities_chroma \
  --collection-name geo_entities
```

Build the DEM patch store from a question-relevant patch list:

```bash
python baselines/build_question_dem_patch_embeddings.py \
  --patches shared_embeddings/dem_patches_gdal_chroma/question_dem_patches.jsonl \
  --persist-directory shared_embeddings/dem_patches_gdal_chroma \
  --collection-name dem_patches \
  --embedding-provider sentence-transformers \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2
```

Run raster RAG:

```bash
python baselines/run_raster_rag.py \
  --input-dir benchmark/qa2/raster_only \
  --input-dir benchmark/qa2/raster_vector \
  --input-dir benchmark/qa2/extended \
  --entity-persist-directory shared_embeddings/vector_entities_chroma \
  --entity-collection-name geo_entities \
  --dem-persist-directory shared_embeddings/dem_patches_gdal_chroma \
  --dem-collection-name dem_patches \
  --embedding-provider sentence-transformers \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --top-k-entities 5 \
  --top-k-dem 5 \
  --spatial-prefilter \
  --llm-provider gemini \
  --model gemini-2.5-flash \
  --output baselines/cache/gemini_rag_raster/raster_rag.jsonl \
  --resume
```

RAG does not execute SQL. It retrieves text/metadata records and asks the model
to answer from the retrieved context. For raster questions, the DEM store
contains metadata and patch summaries rather than full pixel arrays.
