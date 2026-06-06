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

Build the vector entity store:

```bash
cd GS-QA
python baselines/rag/build_vector_entity_embeddings.py \
  --input-dir benchmark/qa2/raster_only \
  --input-dir benchmark/qa2/raster_vector \
  --input-dir benchmark/qa2/extended \
  --persist-directory baselines/shared_embeddings/vector_entities_chroma \
  --collection-name geo_entities
```

Build the DEM patch store from a question-relevant patch list:

```bash
python baselines/build_question_dem_patch_embeddings.py \
  --patches "$SCRATCH_ROOT/needed_dem_patches.jsonl" \
  --persist-directory baselines/shared_embeddings/dem_patches_gdal_chroma \
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
  --entity-persist-directory baselines/shared_embeddings/vector_entities_chroma \
  --entity-collection-name geo_entities \
  --dem-persist-directory baselines/shared_embeddings/dem_patches_gdal_chroma \
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
