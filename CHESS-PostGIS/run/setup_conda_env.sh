#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-chess-postgis}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

cd "$REPO_DIR"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda env update -n "$ENV_NAME" -f environment.yml --prune
else
  conda env create -n "$ENV_NAME" -f environment.yml
fi

conda run -n "$ENV_NAME" python -m ipykernel install --user --name "$ENV_NAME" --display-name "Python ($ENV_NAME)"
conda run -n "$ENV_NAME" python - <<'PY'
import importlib

modules = [
    "langchain_core",
    "langchain",
    "langchain_openai",
    "langchain_google_genai",
    "langchain_chroma",
    "langgraph",
    "psycopg",
    "sqlglot",
    "datasketch",
    "sentence_transformers",
    "faiss",
]

missing = []
for module in modules:
    try:
        importlib.import_module(module)
    except Exception as exc:
        missing.append((module, repr(exc)))

if missing:
    for module, exc in missing:
        print(f"missing_or_broken {module}: {exc}")
    raise SystemExit(1)

print("CHESS conda environment is ready.")
PY
