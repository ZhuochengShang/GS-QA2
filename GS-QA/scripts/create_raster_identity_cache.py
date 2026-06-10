import json
import shutil
from pathlib import Path

source_root = Path("baselines/cache")
target_root = Path("/tmp/raster_identity_cache")
allowlist = {
    line.strip()
    for line in Path("baselines/raster_clean_25_stems.txt").read_text().splitlines()
    if line.strip()
}

if target_root.exists():
    shutil.rmtree(target_root)

count = 0
for query_type in ("raster_only", "raster_vector", "extended"):
    source_dir = source_root / f"gemini_{query_type}"
    target_dir = target_root / f"identity_{query_type}"
    for stem_dir in sorted(source_dir.iterdir()):
        if not stem_dir.is_dir() or stem_dir.name not in allowlist:
            continue
        output_dir = target_dir / stem_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(stem_dir.glob("*.json")):
            if path.name in ("summary.json", "run_summary.json"):
                continue
            record = json.loads(path.read_text())
            gold_rows = (record.get("gold_exec") or {}).get("output", []) or record.get("gold_answers", [])
            record["predicted_exec"] = {"output": gold_rows, "error": ""}
            record["predicted_sql"] = "identity"
            record["query_type"] = query_type
            (output_dir / path.name).write_text(json.dumps(record))
            count += 1

print(count)
