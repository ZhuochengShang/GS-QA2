#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def build_evidence(source_file: str, answer_type: str) -> str:
    parts = []
    if source_file:
        parts.append(f"source={source_file}")
    if answer_type:
        parts.append(f"answer_type={answer_type}")
    return "; ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert raster-only GS-QA JSONL files into CHESS dataset JSON for PostGIS runs."
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing raster JSONL files.")
    parser.add_argument("--output", required=True, help="Output CHESS dataset JSON path.")
    parser.add_argument("--db-id", default="gsqa", help="Database identifier to attach to every task.")
    parser.add_argument(
        "--samples-per-file",
        type=int,
        default=0,
        help="Optional cap per JSONL file. Use 0 for all rows.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    dataset = []
    question_id = 0
    for path in sorted(input_dir.glob("*.jsonl")):
        kept = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                dataset.append(
                    {
                        "question_id": question_id,
                        "db_id": args.db_id,
                        "question": item.get("question", ""),
                        "evidence": build_evidence(path.name, item.get("answer_type", "")),
                        "SQL": item.get("sql", ""),
                    }
                )
                question_id += 1
                kept += 1
                if args.samples_per_file and kept >= args.samples_per_file:
                    break

    output_path.write_text(json.dumps(dataset, indent=2, ensure_ascii=True), encoding="utf-8")
    print(output_path)
    print(f"tasks={len(dataset)}")


if __name__ == "__main__":
    main()
