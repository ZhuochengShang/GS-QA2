# QARV

QARV is a cleaned code-only release bundle for the GS-QA vector/raster
question-answering experiments, including baseline runners, raster evaluation,
CHESS-PostGIS adaptation scripts, and GISCopilot integration code.

## Contents

- `GS-QA/`: GS-QA baseline runners and evaluation scripts.
- `CHESS-PostGIS/`: CHESS adaptation for PostGIS, including raster-aware
  prompts, configs, and collection/evaluation utilities.
- `GISCopilot/`: cleaned GISCopilot source and GS-QA evaluation helpers.

## Data Not Included

Large generated artifacts are intentionally excluded from this code bundle:

- benchmark question folders
- model outputs and result caches
- Chroma vector stores
- SQLite/PostGIS database dumps
- address caches and logs

Release large benchmark/data artifacts separately, for example with Git LFS,
Zenodo, or an institutional data repository. The scripts expect the benchmark
and database paths to be supplied through command-line arguments or environment
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
