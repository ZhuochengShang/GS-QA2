# Qwen Raw Outputs

Raw Qwen3-32B-AWQ outputs used to produce the Qwen evaluation summaries in
`GS-QA/baselines/evaluation`.

Generation setup:

- Model: Qwen3-32B-AWQ
- Thinking: disabled (`chat_template_kwargs.enable_thinking=False`)
- Per-question SQL generation time limit: 360 seconds
- RAG top-k: reduced as needed to fit the 4096-token server context window

Directory layout:

- `vector_text2sql/`: vector-only Text2SQL outputs for T1-T28, copied from
  `/Users/clockorangezoe/Downloads/QWEN_ouput`.
- `raster_text2sql/`: raster-related Text2SQL outputs for `raster_only`,
  `raster_vector`, and `extended`, copied from
  `/Users/clockorangezoe/Downloads/Text2SQL`.
- `rag/`: RAG outputs for vector, raster-only, raster-vector, and extended
  splits, copied from `/Users/clockorangezoe/Downloads/RAG`.
- `manifest.sha256`: SHA-256 checksums for files in this artifact directory.

Derived evaluation files:

- `../qwen_vector/`
- `../qwen_text2sql_raster/`
- `../qwen_raster_rag/`
- `../qwen_combined_summary.csv`
- `../qwen_leaf_metrics.csv`

The paper raster table uses the 25 selected raster-related stems from
`qwen_leaf_metrics.csv`, corresponding to 500 questions total.
