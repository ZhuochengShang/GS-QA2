#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


QA_FILES = {
    "T1": "adjacent_dataset",
    "T2": "amenities-around-specific",
    "T3": "amenities_around_dataset",
    "T4": "amenities_dataset",
    "T5": "compare-closer",
    "T6": "direction_nearest_dataset",
    "T7": "nearest_amenity_dataset",
    "T8": "intersection_dataset",
    "T9": "distance_dataset",
}


TYPE_NOTES = {
    "T1": "nearest entity by requested amenity/category; answer_type=name",
    "T2": "entities of a requested category within 50 meters; answer_type=name_list",
    "T3": "amenity/type values within 100 meters; answer_type=amenity_type_list",
    "T4": "attribute lookup for one entity; answer_type=amenity_type",
    "T5": "compare which candidate is closer to a reference; answer_type=name",
    "T6": "nearest entity of a category in a direction from a reference; answer_type=name",
    "T7": "nearest amenity around a reference; answer_type=name; negative osm_ids may require polygon fallback",
    "T8": "nearest POI to street intersection; use planet_osm_line for streets; answer_type=name",
    "T9": "distance between two entities in meters; answer_type=distance_m",
}


def load_type(dataset_root: Path, ttype: str, limit: int | None = None) -> list[dict]:
    stem = QA_FILES[ttype]
    qa_dir = dataset_root / "llm" / "california_full" / "question-answer"
    rows = []
    with open(qa_dir / f"{stem}.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = [c.strip().lower() for c in reader.fieldnames or []]
        for i, row in enumerate(reader):
            norm = {k.strip().lower(): v for k, v in row.items() if k is not None}
            row_id = norm.get("id") or norm.get("no") or str(i + 1)
            question = norm.get("question", "")
            answer = norm.get("answer", "")
            if ttype == "T9" and "id" not in cols:
                row_id = norm.get("question", str(i + 1))
                question = norm.get("answer", "")
                overflow = row.get(None) or []
                answer = (
                    overflow[0]
                    if isinstance(overflow, list) and overflow
                    else (overflow or "")
                ).strip()
            rows.append({"id": str(row_id).strip(), "question": question, "answer": answer})

    with open(qa_dir / f"{stem}.json", encoding="utf-8") as f:
        meta = json.load(f)

    out = []
    for i, row in enumerate(rows):
        qid = row["id"]
        entities = meta.get(qid) or meta.get(str(i + 1)) or {}
        out.append({**row, "entities": entities})
    return out[:limit] if limit else out


def entity_hint(entities: dict) -> str:
    if not entities:
        return "entities=none"
    parts = []
    for name, data in entities.items():
        if isinstance(data, dict):
            bits = [f"name={name!r}"]
            if "osm_id" in data:
                bits.append(f"osm_id={data['osm_id']}")
            if "latitude" in data and "longitude" in data:
                bits.append(f"lat={data['latitude']}")
                bits.append(f"lon={data['longitude']}")
            parts.append("(" + ", ".join(bits) + ")")
        else:
            parts.append(f"{name}: {data}")
    return "entities=" + "; ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--types", nargs="+", default=sorted(QA_FILES))
    parser.add_argument("--limit-per-type", type=int, default=None)
    args = parser.parse_args()

    dataset = []
    for ttype in args.types:
        for row in load_type(args.dataset_root, ttype, args.limit_per_type):
            evidence = [
                f"dataset=MapQA",
                f"template_type={ttype}",
                TYPE_NOTES[ttype],
                entity_hint(row["entities"]),
                f"gold_answer={row['answer']}",
            ]
            dataset.append({
                "question_id": len(dataset) + 1,
                "db_id": "mapqa",
                "question": row["question"],
                "evidence": " | ".join(evidence),
                "difficulty": ttype,
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    print(f"Wrote {len(dataset)} CHESS MapQA questions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
