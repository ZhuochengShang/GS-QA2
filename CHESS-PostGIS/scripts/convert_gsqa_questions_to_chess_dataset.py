#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Optional


def build_evidence(source_file: str, answer_type: str, gsqa_id: Optional[int] = None) -> str:
    parts = [f"source={source_file}"]
    if gsqa_id is not None:
        parts.append(f"gsqa_id={gsqa_id}")
    parts.append(f"answer_type={answer_type}")
    return "; ".join(parts)


def _split_top_level_args(arg_string: str) -> list[str]:
    args = []
    current = []
    depth = 0
    in_single_quote = False
    i = 0
    while i < len(arg_string):
        ch = arg_string[i]
        if ch == "'" and (i == 0 or arg_string[i - 1] != "\\"):
            in_single_quote = not in_single_quote
            current.append(ch)
        elif not in_single_quote and ch == "(":
            depth += 1
            current.append(ch)
        elif not in_single_quote and ch == ")":
            depth -= 1
            current.append(ch)
        elif not in_single_quote and depth == 0 and ch == ",":
            args.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
        i += 1
    if current:
        args.append("".join(current).strip())
    return args


def _replace_st_dwithin(sql: str) -> str:
    needle = "ST_DWithin("
    while True:
        start = sql.find(needle)
        if start == -1:
            break
        i = start + len(needle)
        depth = 1
        in_single_quote = False
        while i < len(sql) and depth > 0:
            ch = sql[i]
            if ch == "'" and sql[i - 1] != "\\":
                in_single_quote = not in_single_quote
            elif not in_single_quote and ch == "(":
                depth += 1
            elif not in_single_quote and ch == ")":
                depth -= 1
            i += 1
        if depth != 0:
            break
        inside = sql[start + len(needle): i - 1]
        parts = _split_top_level_args(inside)
        if len(parts) != 3:
            break
        replacement = f"(ST_Distance({parts[0]}, {parts[1]}) <= {parts[2]})"
        sql = sql[:start] + replacement + sql[i:]
    return sql


def normalize_postgis_to_sqlite_spatialite(sql: str) -> str:
    if not isinstance(sql, str):
        return ""
    out = sql.strip()
    out = re.sub(r"\s*::\s*geography\b", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\bST_GeomFromText\s*\(", "GeomFromText(", out, flags=re.IGNORECASE)
    out = _replace_st_dwithin(out)
    # Rewrite nearest-neighbor PostGIS operator to distance sort.
    out = re.sub(
        r"(?i)\bORDER\s+BY\s+([A-Za-z_][A-Za-z0-9_.]*)\s*<->\s*(.+?)\s+(ASC|DESC)\b",
        r"ORDER BY ST_Distance(\1, \2) \3",
        out,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return out


def normalize_sql(sql: str, sql_target: str) -> str:
    if not isinstance(sql, str):
        return ""
    if sql_target == "spatialite":
        return normalize_postgis_to_sqlite_spatialite(sql)
    return sql.strip()


def iter_jsonl_items(files: list[Path]):
    for file_path in files:
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield file_path, json.loads(line)


def iter_question_json_items(files: list[Path]):
    for file_path in files:
        yield file_path, json.loads(file_path.read_text(encoding="utf-8"))


def infer_answer_type(item: dict) -> str:
    answer_type = item.get("answer_type")
    if answer_type:
        return answer_type
    answers = item.get("answers")
    if isinstance(answers, list) and answers:
        sample = answers[0]
        if isinstance(sample, dict):
            if "geometry" in sample:
                return "geometry"
            if "poi_name" in sample or "park_name" in sample or "lake_name" in sample:
                return "name"
    return "unknown"


def get_gsqa_question_id(file_path: Path) -> int:
    folder_name = file_path.parent.name.strip()
    if not folder_name.isdigit():
        raise ValueError(f"Cannot infer GS-QA question id from folder name: {file_path.parent.name!r}")
    return int(folder_name)


def question_json_sort_key(path: Path) -> tuple[int, str]:
    try:
        return (get_gsqa_question_id(path), str(path))
    except ValueError:
        return (10**12, str(path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert GS-QA question files into CHESS dataset JSON format."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing GS-QA JSONL files or benchmark folders with question.json files.",
    )
    parser.add_argument(
        "--output_json",
        required=True,
        help="Output CHESS dataset JSON path.",
    )
    parser.add_argument(
        "--db_id",
        default="osm",
        help="Database id to assign for all tasks (default: osm).",
    )
    parser.add_argument(
        "--glob",
        default="*.jsonl",
        help="Glob pattern for input files (default: *.jsonl).",
    )
    parser.add_argument(
        "--source_format",
        choices=("auto", "jsonl", "question_json"),
        default="auto",
        help="Input format. auto uses JSONL files when present, otherwise recursive question.json files.",
    )
    parser.add_argument(
        "--sql_target",
        choices=("postgis", "spatialite"),
        default="spatialite",
        help="Keep PostGIS SQL or rewrite to SQLite/SpatiaLite syntax (default: spatialite).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    source_format = args.source_format
    jsonl_files = sorted(input_dir.glob(args.glob))
    question_json_files = sorted(input_dir.rglob("question.json"), key=question_json_sort_key)
    if source_format == "auto":
        source_format = "jsonl" if jsonl_files else "question_json"

    files = jsonl_files if source_format == "jsonl" else question_json_files
    if not files:
        pattern = args.glob if source_format == "jsonl" else "**/question.json"
        raise FileNotFoundError(f"No files matched {pattern} in {input_dir}")

    tasks = []
    qid = 0
    iterator = iter_jsonl_items(files) if source_format == "jsonl" else iter_question_json_items(files)
    for file_path, item in iterator:
        source_file = str(file_path.relative_to(input_dir))
        gsqa_id = get_gsqa_question_id(file_path) if source_format == "question_json" else None
        question_id = gsqa_id if gsqa_id is not None else qid
        question = item.get("question", "")
        sql = normalize_sql(item.get("sql", ""), args.sql_target)
        answer_type = infer_answer_type(item)
        task = {
            "question_id": question_id,
            "db_id": args.db_id,
            "question": question,
            "evidence": build_evidence(source_file, answer_type, gsqa_id),
            "SQL": sql,
        }
        tasks.append(task)
        qid += 1

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(tasks)} tasks to {output_json}")


if __name__ == "__main__":
    main()
