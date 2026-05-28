#!/usr/bin/env python3
import argparse
import json
import os
import glob
import re
import sys
import sqlite3
from collections import defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from database_utils.spatial import register_spatial_functions, parse_geometry_to_lon_lat


def parse_answer_type(evidence: str) -> str:
    if not evidence:
        return ""
    m = re.search(r"answer_type=([^;\s]+)", evidence)
    return m.group(1).strip() if m else ""


def parse_source(evidence: str) -> str:
    if not evidence:
        return "unknown"
    m = re.search(r"source=([^;\s]+)", evidence)
    src = m.group(1).strip() if m else "unknown"
    if src.endswith('.jsonl'):
        src = src[:-6]
    return src


def _env_truthy(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _resolve_spatialite_extension_path() -> str:
    path = os.getenv("SPATIALITE_EXTENSION_PATH", "").strip()
    if path:
        return path
    candidates = []
    conda_prefix = os.getenv("CONDA_PREFIX")
    if conda_prefix:
        candidates.extend(sorted(glob.glob(f"{conda_prefix}/lib/mod_spatialite*.dylib")))
        candidates.extend(sorted(glob.glob(f"{conda_prefix}/lib/mod_spatialite*.so")))
    candidates.extend([
        "/opt/homebrew/lib/mod_spatialite.dylib",
        "/opt/homebrew/lib/mod_spatialite.8.dylib",
        "/usr/local/lib/mod_spatialite.dylib",
        "/usr/local/lib/mod_spatialite.so",
    ])
    for c in candidates:
        if os.path.exists(c):
            return c
    return ""


def _connect_sqlite(sqlite_db: str) -> sqlite3.Connection:
    conn = sqlite3.connect(sqlite_db)
    if _env_truthy("SPATIALITE_ENABLED", "0"):
        ext_path = _resolve_spatialite_extension_path()
        if not ext_path:
            conn.close()
            raise RuntimeError(
                "SPATIALITE_ENABLED=1 but no extension path was found. "
                "Set SPATIALITE_EXTENSION_PATH explicitly."
            )
        conn.enable_load_extension(True)
        try:
            conn.load_extension(ext_path)
        finally:
            conn.enable_load_extension(False)
    # Always provide PostGIS-like helpers for ST_Distance/GeomFromWKB.
    # If SpatiaLite is present, this still keeps behavior consistent for CHESS SQL.
    register_spatial_functions(conn)
    return conn


def build_ids(dataset):
    counters = defaultdict(int)
    id_map = {}
    for item in dataset:
        qid = int(item["question_id"])
        source = parse_source(item.get("evidence", ""))
        idx = counters[source]
        counters[source] += 1
        id_map[qid] = f"{source}-{idx}"
    return id_map


def load_llm_metrics(metrics_path: Path):
    # Aggregate per question_id from -llm_metrics.jsonl if present
    by_qid = defaultdict(lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "latency_ms": 0.0})
    if not metrics_path.exists():
        return by_qid
    for line in metrics_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            qid = int(d.get("question_id"))
            by_qid[qid]["prompt_tokens"] += int(d.get("prompt_tokens", 0) or 0)
            by_qid[qid]["completion_tokens"] += int(d.get("completion_tokens", 0) or 0)
            by_qid[qid]["total_tokens"] += int(d.get("total_tokens", 0) or 0)
            by_qid[qid]["latency_ms"] += float(d.get("latency_ms", 0.0) or 0.0)
        except Exception:
            continue
    return by_qid


def sql_from_prediction(v):
    if isinstance(v, str):
        return v.split("\t----- bird -----\t")[0].strip()
    return ""


def _normalize_jsonable(v):
    if isinstance(v, bytes):
        coords = parse_geometry_to_lon_lat(v)
        if coords is not None:
            return f"POINT({coords[0]} {coords[1]})"
        return v.hex()
    if isinstance(v, memoryview):
        raw = v.tobytes()
        coords = parse_geometry_to_lon_lat(raw)
        if coords is not None:
            return f"POINT({coords[0]} {coords[1]})"
        return raw.hex()
    if isinstance(v, str):
        coords = parse_geometry_to_lon_lat(v)
        if coords is not None:
            return f"POINT({coords[0]} {coords[1]})"
    return v


def _normalize_row(d):
    return {k: _normalize_jsonable(v) for k, v in d.items()}


def rows_to_text(rows, answer_type):
    if not rows:
        return ""

    def first_non_null(d, keys):
        for k in keys:
            if k in d and d[k] is not None and str(d[k]).strip() != "":
                return d[k]
        return None

    if answer_type in {"count", "distance", "length", "area"}:
        # Prefer aggregate-like fields
        d = rows[0]
        for k in ["count", "distance", "length", "area", "COUNT(*)", "count(*)"]:
            if k in d and d[k] is not None:
                return str(d[k])
        return str(next(iter(d.values())))

    if answer_type in {"name", "multihop_attribute"}:
        names = []
        for d in rows:
            v = first_non_null(d, ["poi_name", "park_name", "road_name", "region_name", "name", "official_name"])
            if v is not None:
                names.append(str(v))
        names = list(dict.fromkeys(names))
        return "\n".join(names[:10]) if names else json.dumps(_normalize_row(rows[0]), ensure_ascii=True)

    if answer_type == "loc":
        out = []
        for d in rows[:10]:
            addr = []
            for k in ["addr_housenumber", "addr_street", "addr_city", "addr_state", "addr_postcode"]:
                if d.get(k):
                    addr.append(str(d[k]))
            if addr:
                out.append(", ".join(addr))
            elif d.get("poi_name"):
                out.append(str(d["poi_name"]))
        return "\n".join(out) if out else json.dumps(_normalize_row(rows[0]), ensure_ascii=True)

    if answer_type == "angle":
        d = rows[0]
        for k in ["angle", "azimuth", "bearing"]:
            if k in d and d[k] is not None:
                return str(d[k])
        return json.dumps(_normalize_row(d), ensure_ascii=True)

    # Fallback: one line per row (compact)
    return "\n".join(json.dumps(_normalize_row(r), ensure_ascii=True) for r in rows[:5])


def text_to_json_block(text, answer_type):
    payload = {}
    t = text.strip()
    if not t:
        return "```json\n{}\n```"

    if answer_type in {"count", "distance", "length", "area", "angle"}:
        m = re.search(r"-?\d+(?:\.\d+)?", t)
        val = float(m.group(0)) if m else None
        if answer_type == "count" and val is not None:
            payload["count"] = int(round(val))
        elif answer_type == "distance" and val is not None:
            payload["distance"] = int(round(val))
        elif answer_type == "length" and val is not None:
            payload["length"] = int(round(val))
        elif answer_type == "area" and val is not None:
            payload["area"] = int(round(val))
        elif answer_type == "angle" and val is not None:
            payload["azimuth_angle"] = int(round(val))
    elif answer_type in {"name", "multihop_attribute"}:
        first = t.splitlines()[0].strip()
        payload["name"] = first
    elif answer_type == "loc":
        first = t.splitlines()[0].strip()
        payload["address"] = first
    else:
        payload["name"] = t.splitlines()[0].strip()

    return "```json\n" + json.dumps(payload, ensure_ascii=True, indent=2) + "\n```"


def main():
    ap = argparse.ArgumentParser(description="Export CHESS outputs into GS-QA baseline-like JSON files")
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sqlite-db", required=True)
    ap.add_argument("--run-dir", required=True, help="Directory where output JSON files will be written")
    ap.add_argument("--prefix", default="chess")
    args = ap.parse_args()

    predictions = json.loads(Path(args.predictions).read_text())
    dataset = json.loads(Path(args.dataset).read_text())
    id_map = build_ids(dataset)
    qmeta = {int(x["question_id"]): x for x in dataset}

    run_dir = Path(args.run_dir)
    metrics = load_llm_metrics(run_dir / "-llm_metrics.jsonl")

    conn = _connect_sqlite(args.sqlite_db)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    out_sql = []
    out_ans = []
    out_json_ans = []

    for qid_str in sorted(predictions.keys(), key=lambda s: int(s)):
        qid = int(qid_str)
        meta = qmeta.get(qid, {})
        aid = id_map.get(qid, f"q-{qid}")
        answer_type = parse_answer_type(meta.get("evidence", ""))
        sql = sql_from_prediction(predictions[qid_str])

        sql_block = f"```sql\n{sql}\n```" if sql else ""
        status = "ok"
        reason = ""

        answer_text = ""
        rows = []
        if sql:
            try:
                cur.execute(sql)
                rows = [dict(r) for r in cur.fetchmany(100)]
                answer_text = rows_to_text(rows, answer_type)
                if len(rows) == 0:
                    status = "empty"
                    reason = "zero_rows"
                elif not answer_text.strip():
                    status = "empty"
                    reason = "unrenderable_rows"
            except Exception as e:
                status = "empty"
                reason = f"sql_error: {e}"
                answer_text = ""
        else:
            status = "empty"
            reason = "empty_prediction"

        out_sql.append({
            "id": aid,
            "content": sql_block,
            "status": status,
            "reason": reason,
        })

        out_ans.append({
            "id": aid,
            "content": answer_text,
            "status": status,
            "reason": reason,
        })

        m = metrics.get(qid, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "latency_ms": 0.0})
        out_json_ans.append({
            "id": aid,
            "content": text_to_json_block(answer_text, answer_type),
            "status": status,
            "reason": reason,
            "run_seconds": round(float(m.get("latency_ms", 0.0)) / 1000.0, 3),
            "prompt_tokens": int(m.get("prompt_tokens", 0)),
            "completion_tokens": int(m.get("completion_tokens", 0)),
            "total_tokens": int(m.get("total_tokens", 0)),
        })

    conn.close()

    (run_dir / f"text2sql_output_{args.prefix}.json").write_text(json.dumps(out_sql, indent=2, ensure_ascii=True))
    (run_dir / f"text2sql_answers_{args.prefix}.json").write_text(json.dumps(out_ans, indent=2, ensure_ascii=True))
    (run_dir / f"text2sql_json_answers_{args.prefix}.json").write_text(json.dumps(out_json_ans, indent=2, ensure_ascii=True))

    print("wrote:")
    print(run_dir / f"text2sql_output_{args.prefix}.json")
    print(run_dir / f"text2sql_answers_{args.prefix}.json")
    print(run_dir / f"text2sql_json_answers_{args.prefix}.json")


if __name__ == "__main__":
    main()
