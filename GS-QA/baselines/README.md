# Baselines Layout

This directory separates runnable baseline code from model outputs and
evaluation artifacts.

## Runners

- `baselines.py`: vector-only direct, Text2SQL, and RAG runner for T1-T28.
- `text2sql/`: raster Text2SQL runner and Text2SQL result collection helpers.
- `rag/`: raster RAG runner, embedding builders, and model-specific RAG wrappers.
- `chess/`: CHESS/PostGIS collection and evaluation helpers. Local CHESS runtime
  folders belong under `chess/runtime/` and are ignored by git.
- `evaluators/`: vector, raster, and table-formatting evaluators.

Thin compatibility wrappers remain at the older script paths, for example
`baselines/evaluate_raster.py` and `baselines/run_raster_text2sql.py`.

## Outputs

Evaluation outputs are grouped by model:

- `evaluation/qwen/raw_outputs/all/`: organized raw Qwen generation artifacts.
- `evaluation/qwen/vector/`: Qwen vector-only evaluation summaries.
- `evaluation/qwen/text2sql/raster/`: Qwen raster Text2SQL evaluation summaries.
- `evaluation/qwen/rag/raster/`: Qwen raster RAG evaluation summaries.
- `evaluation/qwen/tables/`: combined CSV summaries and paper table metrics.
- `evaluation/gemini/`: reserved for Gemini outputs.

The copy-paste LaTeX table file is local-only and ignored by git.

