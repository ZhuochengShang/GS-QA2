#!/usr/bin/env python3
"""Cluster-friendly raster Text2SQL runner for GS-QA.

Directory layout expected:
  benchmark/q&a-2/
    raster_only/     *.jsonl
    raster_vector/   *.jsonl
    extended/        *.jsonl

Output layout produced:
  baselines/cache/{provider}_{query_type}/
    prompt.txt
    run_summary.json                 <- aggregate across all stems
    {stem}/                          <- one dir per JSONL file
      {stem}-0.json                  <- one file per question
      {stem}-1.json
      ...
      summary.json                   <- aggregate for this stem
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import statistics
import time
import urllib.error
import urllib.request
from decimal import Decimal
from pathlib import Path
from typing import Any

import psycopg
from openai import OpenAI


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = """You convert geospatial raster questions into PostGIS SQL.

DATABASE:
- Table: {dem_table}
- Column: rast (PostGIS raster)

RASTER DATASET:
- {dem_table}.rast stores ASTER GDEM Version 3 elevation values
- DEM resolution: 1 arc second (about 30 m at the equator)
- Horizontal CRS: EPSG:4326 / WGS84
- Elevation reference: EGM96 geoid
- Coverage: land areas from 83N to 83S
- The raster is tiled; example database metadata shows 256x256 tiles, 1 band, SRID 4326
- ASTER GDEM tiles may overlap by 1 pixel at tile edges
- The dataset may contain artifacts or anomalies

VECTOR GEOMETRY SOURCES:
- pois: points of interest such as hospitals, restaurants, museums, amenities
- lakes: lakes, rivers, waterbodies
- parks: parks, gardens
- roads: roads, walkways, highways
- regions: cities, states, counties, administrative areas

OSM VECTOR TABLE SCHEMA:

Table 1: pois
contains the points of interest
Schema:
- id: unique identifier
- geometry: geography type represents the shape and position on earth
- poi_name: name of the poi
- wikidata: unique identifier reference to wikidata
- wikipedia: unique name reference to wikipedia page
- addr_state: the state where the poi is located
- addr_city: the city where the poi is located
- cuisine: type of cuisine associated with restaurant pois
- leisure: type of leisure
- tourism: type of tourism
- takeaway: indicates if a restaurant offers takeaway
- drive_through: indicates if a restaurant offers drive through
- museum: type of museum
- healthcare: type of healthcare service
- outdoor_seating: indicates if an amenity has outdoor seating
- emergency: indicates if a healthcare service provides emergency service
- restaurant: attribute related to restaurants
- amenity: type of amenity provided

Table 2: lakes
contains lakes, rivers, waterbodies, etc.
Schema:
- id: unique identifier
- geometry: geography type represents the shape and position on earth
- lake_name: name of the lake
- wikidata: unique identifier reference to wikidata
- wikipedia: unique name reference to wikipedia page
- addr_country: the country where the lake is located
- addr_state: the state where the lake is located
- addr_county: the county where the lake is located
- addr_city: the city where the lake is located
- addr_postcode: postcode where the lake is located
- addr_street: street where the lake is located
- addr_housenumber: house number specific to this lake
- waterway: type of waterway
- water: type of waterbody

Table 3: parks
contains parks, gardens, etc.
Schema:
- id: unique identifier
- geometry: geography type represents the shape and position on earth
- park_name: name of the park
- wikidata: unique identifier reference to wikidata
- wikipedia: unique name reference to wikipedia page
- leisure: type of leisure
- park: type of park
- tourism: type of tourism

Table 4: roads
contains roads, walkways, etc.
Schema:
- id: unique identifier
- geometry: geography type represents the shape and position on earth
- road_name: name of the road
- wikidata: unique identifier reference to wikidata
- wikipedia: unique name reference to wikipedia page
- highway: attribute associated with roads of type highway
- sidewalk: attribute associated with roads of type sidewalk
- foot: attribute associated with roads of type foot
- bicycle: attribute associated with roads of type bicycle
- cycleway: attribute associated with roads of type cycleway

Table 5: regions
contains administrative region boundaries, like cities and states, etc.
Schema:
- id: integer
- geometry: geography type represents the shape and position on earth
- region_name: name of the region
- border_type: the type of border
- wikidata: unique identifier reference to wikidata
- wikipedia: unique name reference to wikipedia page

INPUT:
You are given:
1. A natural language question
2. Structured context (may include WKT geometry and conditions)

RULES:
- Use {dem_table} for all raster and terrain computations.
- Use pois, lakes, parks, roads, or regions only to obtain geometry referenced in the question.
- Do not use vector tables for the final numeric terrain answer except as geometry sources or spatial filters.
- Never invent coordinates or external data.
- If structured context provides WKT, use it directly.
- If geometry is required and no WKT is provided, derive it from the appropriate vector table.
- Resolve named entities by type:
  - POIs and amenities -> pois
  - rivers, lakes, waterbodies -> lakes
  - parks and gardens -> parks
  - roads and highways -> roads
  - cities, states, counties, administrative places -> regions
- Queries must match the provided table names and columns.
- Queries must be compatible with PostGIS.
- Use only columns explicitly listed for each table.
- Do not use `addr_state` or `addr_city` on `regions`; those columns do not exist in `regions`.
- To disambiguate places in `regions`, use valid `regions` columns such as `region_name`,
  `border_type`, `wikidata`, `wikipedia`, or spatial intersection with another `regions` row.
- All vector `geometry` columns are of type `geography`.
- For raster operations, cast vector geography to geometry with `geometry::geometry` when needed.
- Use `ILIKE` with `%` for flexible string matching.
- Do not include geometry in the final output unless the question explicitly asks for it.
- For comparison questions (which of two places is higher, steeper, etc.), return the name of
  the winning entity as a string column alongside the numeric values.
- Output only one SQL block: ```sql ... ```

ALWAYS:
- Use ST_Clip before aggregation if geometry is provided.
- Use ST_Intersects to filter raster tiles.
- Avoid unnecessary joins.
- Return a single scalar result when possible.
- If a named feature must be resolved first, isolate that lookup in a CTE and then apply
  raster analysis against the resulting geometry.
- Treat band 1 as elevation.
- Do not assume the DEM already stores derived terrain products like slope.
- If slope, aspect, or steepness is requested, derive it from the DEM before summarizing.
"""


# ---------------------------------------------------------------------------
# Per-query-type tolerances
# ---------------------------------------------------------------------------

ANSWER_TYPE_TOLERANCES: dict[str, dict[str, float]] = {
    "mean elevation":       {"abs": 5.0,  "rel": 1e-6},
    "coverage":             {"abs": 5.0,  "rel": 1e-6},
    "average slope":        {"abs": 5.0,  "rel": 1e-6},
    "minimum slope":        {"abs": 5.0,  "rel": 1e-6},
    "elevation":            {"abs": 10.0, "rel": 1e-6},
    "elevation difference": {"abs": 10.0, "rel": 1e-6},
    "slope":                {"abs": 10.0, "rel": 1e-6},
    "aspect":               {"abs": 10.0, "rel": 1e-6},
    "distance":             {"abs": 1e-6, "rel": 0.05},
    "area":                 {"abs": 1e-6, "rel": 0.05},
    "count":                {"abs": 1e-6, "rel": 0.05},
    "ruggedness":           {"abs": 1e-6, "rel": 0.05},
}

QUERY_TYPES = ("raster_only", "raster_vector", "extended")


def get_tolerances(
    answer_type: str | None,
    default_abs: float,
    default_rel: float,
) -> tuple[float, float]:
    key = (answer_type or "").lower()
    for name, tols in ANSWER_TYPE_TOLERANCES.items():
        if name in key:
            return tols["abs"], tols["rel"]
    return default_abs, default_rel


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run raster Text2SQL over benchmark subfolders."
    )
    parser.add_argument(
        "--benchmark-dir",
        default="benchmark/qa2",
        help="Root benchmark dir containing raster_only/, raster_vector/, extended/.",
    )
    parser.add_argument(
        "--cache-dir",
        default="baselines/cache",
        help="Root output dir; results go to {cache_dir}/{provider}_{query_type}/.",
    )
    parser.add_argument(
        "--query-types",
        nargs="+",
        choices=QUERY_TYPES,
        default=list(QUERY_TYPES),
        help="Which subfolders to process (default: all three).",
    )
    parser.add_argument(
        "--samples-per-file",
        type=int,
        default=0,
        help="Max questions per JSONL file. 0 = all.",
    )
    parser.add_argument("--model",    default="gemini-2.5-flash")
    parser.add_argument(
        "--provider",
        choices=("openai", "gemini"),
        default="gemini",
    )
    parser.add_argument(
        "--openai-base-url",
        default=os.environ.get("OPENAI_BASE_URL"),
        help="Optional OpenAI-compatible base URL, e.g. http://localhost:8000/v1 for local Qwen.",
    )
    parser.add_argument("--dem-table",    default="public.dem_us")
    parser.add_argument("--db-host",      default="localhost")
    parser.add_argument("--db-name",      default="gsqa")
    parser.add_argument("--db-user",      default="zshan011")
    parser.add_argument("--db-password",  default="")
    parser.add_argument("--db-port",      type=int, default=5432)
    parser.add_argument(
        "--numeric-abs-tolerance",
        type=float,
        default=10.0,
        help="Fallback absolute tolerance when answer type is unrecognised.",
    )
    parser.add_argument(
        "--numeric-rel-tolerance",
        type=float,
        default=0.05,
        help="Fallback relative tolerance when answer type is unrecognised.",
    )
    parser.add_argument(
        "--generation-timeout",
        type=int,
        default=360,
        help="Per-question generation timeout in seconds (default 360 = 6 min).",
    )
    parser.add_argument(
        "--execution-timeout",
        type=int,
        default=180,
        help="Per-question SQL execution timeout in seconds (default 180 = 3 min).",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Stop after generation + execution; skip compare/summary.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of parallel workers per stem. Each worker opens its own "
            "DB connection. 1 = sequential (default). "
            "Recommended: 4-8 for I/O-bound Gemini calls."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# I/O — per-question files
# ---------------------------------------------------------------------------

def question_path(stem_dir: Path, question_id: str) -> Path:
    return stem_dir / f"{question_id}.json"


def load_question_result(stem_dir: Path, question_id: str) -> dict[str, Any] | None:
    p = question_path(stem_dir, question_id)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def save_question_result(stem_dir: Path, record: dict[str, Any]) -> None:
    stem_dir.mkdir(parents=True, exist_ok=True)
    p = question_path(stem_dir, record["id"])
    p.write_text(json.dumps(normalize_value(record), indent=2), encoding="utf-8")


def load_questions(input_dir: Path, samples_per_file: int) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as fh:
            kept = 0
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record                 = json.loads(line)
                record["source_file"]  = path.name
                record["source_stem"]  = path.stem
                record["id"]           = f"{path.stem}-{kept}"
                questions.append(record)
                kept += 1
                if samples_per_file and kept >= samples_per_file:
                    break
    return questions


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def compact_question_entities(
    entities: dict[str, Any],
    max_wkt_chars: int = 400,
) -> dict[str, Any]:
    compacted: dict[str, Any] = {}
    for key, value in entities.items():
        if not isinstance(value, dict):
            compacted[key] = value
            continue
        item: dict[str, Any] = {}
        for field in (
            "main_category", "sub_category", "display_name",
            "distance", "elevation", "area", "length", "count", "condition",
        ):
            if field in value:
                item[field] = value[field]
        geo_wkt = value.get("geo_wkt")
        if isinstance(geo_wkt, str):
            if len(geo_wkt) <= max_wkt_chars or geo_wkt.startswith("POINT"):
                item["geo_wkt"] = geo_wkt
            else:
                item["geo_wkt_omitted"] = f"omitted_large_wkt_{len(geo_wkt)}_chars"
        poi = value.get("poi")
        if isinstance(poi, dict):
            item["poi"] = {
                k: poi[k]
                for k in ("id", "poi_name", "amenity", "addr_city", "addr_state")
                if k in poi
            }
        compacted[key] = item if item else value
    return compacted


def build_user_prompt(question: dict[str, Any]) -> str:
    ctx = {
        "answer_type":       question.get("answer_type"),
        "question_entities": compact_question_entities(
            question.get("question_entities", {})
        ),
    }
    return (
        f"Question:\n{question['question']}\n\n"
        f"Structured context:\n{json.dumps(ctx, indent=2, ensure_ascii=True)}\n\n"
        "Use only the geometries and constraints shown above. "
        "If a very large WKT is omitted, derive geometry from the named feature "
        "using the appropriate vector table instead of inventing coordinates.\n"
    )


# ---------------------------------------------------------------------------
# LLM invocation
# ---------------------------------------------------------------------------

def invoke_openai(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    )
    usage = response.usage
    return {
        "text": response.choices[0].message.content or "",
        "usage": {
            "prompt_tokens":     getattr(usage, "prompt_tokens",     None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens":      getattr(usage, "total_tokens",      None),
        },
    }


def invoke_gemini(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> dict[str, Any]:
    model_name   = model if model.startswith("models/") else f"models/{model}"
    request_body = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents":          [{"parts": [{"text": user_prompt}]}],
        "generationConfig":  {"temperature": 0},
    }
    req = urllib.request.Request(
        url=(
            f"https://generativelanguage.googleapis.com/v1beta/"
            f"{model_name}:generateContent?key={api_key}"
        ),
        data=json.dumps(request_body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"Gemini HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Gemini URLError: {exc}") from exc

    texts = [
        part["text"]
        for c in payload.get("candidates", [])
        for part in c.get("content", {}).get("parts", [])
        if part.get("text")
    ]
    meta = payload.get("usageMetadata", {})
    if texts:
        return {
            "text": "\n".join(texts),
            "usage": {
                "prompt_tokens":     meta.get("promptTokenCount"),
                "completion_tokens": meta.get("candidatesTokenCount"),
                "total_tokens":      meta.get("totalTokenCount"),
            },
        }
    if payload.get("promptFeedback"):
        raise RuntimeError(f"Gemini no candidates: {payload['promptFeedback']}")
    raise RuntimeError(f"Gemini no candidates: {payload}")


def invoke_model(
    provider: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    openai_client: OpenAI | None = None,
    gemini_api_key: str | None = None,
) -> dict[str, Any]:
    if provider == "openai":
        if openai_client is None:
            raise RuntimeError("OpenAI client required.")
        return invoke_openai(openai_client, model, system_prompt, user_prompt)
    if provider == "gemini":
        if not gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY required.")
        return invoke_gemini(gemini_api_key, model, system_prompt, user_prompt)
    raise RuntimeError(f"Unknown provider: {provider}")


def invoke_model_with_timeout(
    timeout_seconds: int,
    **kwargs: Any,
) -> dict[str, Any]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(invoke_model, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            return {
                "text":  "",
                "usage": {},
                "generation_error": f"generation_timeout_{timeout_seconds}s",
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "text":  "",
                "usage": {},
                "generation_error": str(exc),
            }


# ---------------------------------------------------------------------------
# SQL helpers
# ---------------------------------------------------------------------------

def clean_sql_candidate(text: str) -> str:
    sql = text.strip()
    sql = re.sub(r"^\s*```(?:sql)?\s*", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\s*```\s*$", "", sql)
    sql = sql.strip().strip("`").strip()
    sql = re.sub(r"^\s*sql\s*", "", sql, flags=re.IGNORECASE)
    return sql.strip()


def extract_sql_blocks(text: str) -> list[str]:
    matches = re.findall(r"```[\s]*sql(.*?)```", text, re.DOTALL | re.IGNORECASE)
    blocks  = [clean_sql_candidate(m) for m in matches if clean_sql_candidate(m)]
    if blocks:
        return blocks
    cleaned = clean_sql_candidate(text)
    match   = re.search(r"(?is)\b(select|with)\b", cleaned)
    if not match:
        return []
    stmt  = cleaned[match.start():]
    fence = re.search(r"\n```", stmt)
    if fence:
        stmt = stmt[:fence.start()]
    semi = re.search(r";", stmt)
    if semi:
        stmt = stmt[:semi.end()]
    return [stmt.strip()] if stmt.strip() else []


def run_sql(
    conn: psycopg.Connection,
    sql: str,
    timeout_ms: int = 180_000,
) -> dict[str, Any]:
    cur = conn.cursor()
    cur.execute(f"SET statement_timeout = {int(timeout_ms)}")
    try:
        cur.execute(sql)
        rows    = cur.fetchmany(size=50)
        columns = [d.name for d in cur.description] if cur.description else []
        output  = [dict(zip(columns, row)) for row in rows]
        cur.close()
        return {"output": output, "error": ""}
    except Exception as exc:  # noqa: BLE001
        conn.rollback()
        cur.close()
        return {"output": [], "error": str(exc)}


def execute_sql_candidates(
    conn: psycopg.Connection,
    sql_candidates: list[str],
    timeout_ms: int = 180_000,
) -> tuple[str, dict[str, Any]]:
    if not sql_candidates:
        return "", {"output": [], "error": "No SQL block found in model output."}
    first_candidate = sql_candidates[0]
    first_result: dict[str, Any] | None = None
    for sql in sql_candidates:
        result = run_sql(conn, sql, timeout_ms=timeout_ms)
        if first_result is None:
            first_result = result
        if not result["error"]:
            return sql, result
    return first_candidate, first_result or {"output": [], "error": "No candidates executed."}


# ---------------------------------------------------------------------------
# Value helpers
# ---------------------------------------------------------------------------

def normalize_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return round(float(value), 6)
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {k: normalize_value(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [normalize_value(i) for i in value]
    return value


def extract_numbers(value: Any) -> list[float]:
    nums: list[float] = []
    if isinstance(value, (int, float)):
        nums.append(float(value))
    elif isinstance(value, dict):
        for v in value.values():
            nums.extend(extract_numbers(v))
    elif isinstance(value, list):
        for v in value:
            nums.extend(extract_numbers(v))
    return nums


def canonicalize_row(row: dict[str, Any]) -> str:
    return json.dumps(normalize_value(row), sort_keys=True, ensure_ascii=True)


def normalize_text(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())


# ---------------------------------------------------------------------------
# Field pickers
# ---------------------------------------------------------------------------

def collect_string_fields(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        for k, v in row.items():
            if isinstance(v, str) and v.strip():
                out.setdefault(k, []).append(v.strip())
    return out


def collect_numeric_fields(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        for k, v in row.items():
            if isinstance(v, (int, float)):
                out.setdefault(k, []).append(float(v))
    return out


def pick_primary_text_field(rows: list[dict[str, Any]]) -> str | None:
    fields = collect_string_fields(rows)
    if not fields:
        return None
    for f in (
        "answer", "answer_name", "higher_poi", "steeper_poi",
        "name", "poi_name", "road_name", "park_name", "lake_name", "region_name",
    ):
        if f in fields:
            return f
    return sorted(fields)[0]


def pick_primary_numeric_field(
    rows: list[dict[str, Any]],
    answer_type: str | None,
) -> str | None:
    fields = collect_numeric_fields(rows)
    if not fields:
        return None
    for f in (
        "elevation", "elevation_m", "elevation_difference_m",
        "mean_elevation", "max_elevation", "min_elevation", "elevation_range",
        "slope_degrees", "slope", "aspect_degrees", "aspect",
        "coverage", "coverage_percent", "percent", "percentage",
        "distance_m", "area", "count", "ruggedness",
    ):
        if f in fields:
            return f
    key = (answer_type or "").lower().split()[0] if answer_type else ""
    for f in fields:
        if key and key in f.lower():
            return f
    return sorted(fields)[0]


# ---------------------------------------------------------------------------
# Metric routing
# ---------------------------------------------------------------------------

def infer_metric_name(question: dict[str, Any]) -> str:
    stem        = str(question.get("source_stem",  "")).lower()
    answer_type = str(question.get("answer_type",  "")).lower()
    if "compare" in stem:
        return "primary_text_exact"
    if answer_type in ("yes-no", "name", "name+elevation", "name+slope"):
        return "primary_text_exact"
    if "name" in answer_type:
        return "primary_text_exact"
    if any(t in answer_type for t in (
        "elevation", "slope", "aspect", "coverage",
        "distance", "area", "percent", "ruggedness", "count",
    )):
        return "primary_numeric_tolerance"
    return "rowset_f1"


def is_scalar_numeric_result(rows: list[dict[str, Any]]) -> bool:
    if len(rows) != 1:
        return False
    vals = list(rows[0].values())
    return len(vals) == 1 and isinstance(vals[0], (int, float))


# ---------------------------------------------------------------------------
# Comparison functions
# ---------------------------------------------------------------------------

def compare_scalar_results(
    predicted: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    abs_tolerance: float,
    rel_tolerance: float,
) -> dict[str, Any]:
    pv  = float(next(iter(predicted[0].values())))
    gv  = float(next(iter(gold[0].values())))
    ae  = abs(pv - gv)
    re_ = ae / max(abs(gv), 1e-12)
    return {
        "metric_family":    "scalar_numeric",
        "predicted_value":  pv,
        "gold_value":       gv,
        "absolute_error":   round(ae,  6),
        "relative_error":   round(re_, 6),
        "within_tolerance": ae <= abs_tolerance or re_ <= rel_tolerance,
    }


def compare_rowset_results(
    predicted: list[dict[str, Any]],
    gold: list[dict[str, Any]],
) -> dict[str, Any]:
    pk  = {canonicalize_row(r) for r in predicted}
    gk  = {canonicalize_row(r) for r in gold}
    tp  = len(pk & gk)
    pre = tp / len(pk) if pk else 0.0
    rec = tp / len(gk) if gk else 0.0
    f1  = 2 * pre * rec / (pre + rec) if pre + rec else 0.0
    return {
        "metric_family":  "rowset",
        "true_positives": tp,
        "precision":      round(pre, 6),
        "recall":         round(rec, 6),
        "f1":             round(f1,  6),
    }


def compare_primary_text_results(
    predicted: list[dict[str, Any]],
    gold: list[dict[str, Any]],
) -> dict[str, Any]:
    gf    = pick_primary_text_field(gold)
    pf    = pick_primary_text_field(predicted)
    gv    = gold[0].get(gf)      if gold      and gf else None
    pv    = predicted[0].get(pf) if predicted and pf else None
    exact = (
        normalize_text(pv) == normalize_text(gv)
        if gv is not None and pv is not None else False
    )
    return {
        "metric_family":   "primary_text",
        "gold_field":      gf, "predicted_field": pf,
        "gold_value":      gv, "predicted_value": pv,
        "exact_match":     exact,
    }


def compare_primary_numeric_results(
    predicted: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    answer_type: str | None,
    abs_tolerance: float,
    rel_tolerance: float,
) -> dict[str, Any]:
    gf = pick_primary_numeric_field(gold,      answer_type=answer_type)
    pf = pick_primary_numeric_field(predicted, answer_type=answer_type)
    gv = gold[0].get(gf)      if gold      and gf else None
    pv = predicted[0].get(pf) if predicted and pf else None
    if not isinstance(gv, (int, float)) or not isinstance(pv, (int, float)):
        return {
            "metric_family": "primary_numeric",
            "gold_field": gf,   "predicted_field": pf,
            "gold_value": gv,   "predicted_value": pv,
            "absolute_error": None, "relative_error": None,
            "within_tolerance": False,
        }
    ae  = abs(float(pv) - float(gv))
    re_ = ae / max(abs(float(gv)), 1e-12)
    return {
        "metric_family":    "primary_numeric",
        "gold_field":       gf,          "predicted_field":  pf,
        "gold_value":       float(gv),   "predicted_value":  float(pv),
        "absolute_error":   round(ae,  6),
        "relative_error":   round(re_, 6),
        "within_tolerance": ae <= abs_tolerance or re_ <= rel_tolerance,
    }


def compare_results(
    question: dict[str, Any],
    predicted: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    answer_type: str | None,
    abs_tolerance: float,
    rel_tolerance: float,
) -> dict[str, Any]:
    pn     = normalize_value(predicted)
    gn     = normalize_value(gold)
    deltas = [round(p - g, 6) for p, g in zip(extract_numbers(pn), extract_numbers(gn))]
    tol_abs, tol_rel = get_tolerances(answer_type, abs_tolerance, rel_tolerance)

    metrics: dict[str, Any] = {
        "answer_type":         answer_type,
        "selected_metric":     infer_metric_name(question),
        "exact_match":         pn == gn,
        "predicted_row_count": len(predicted),
        "gold_row_count":      len(gold),
        "numeric_deltas":      deltas,
        "abs_tolerance_used":  tol_abs,
        "rel_tolerance_used":  tol_rel,
    }
    if deltas:
        abs_d = [abs(d) for d in deltas]
        metrics["max_abs_numeric_delta"]  = max(abs_d)
        metrics["mean_abs_numeric_delta"] = statistics.mean(abs_d)

    sel = metrics["selected_metric"]
    if predicted and gold and sel == "primary_text_exact":
        metrics["typed_metrics"] = compare_primary_text_results(predicted, gold)
    elif predicted and gold and sel == "primary_numeric_tolerance":
        metrics["typed_metrics"] = compare_primary_numeric_results(
            predicted, gold, answer_type=answer_type,
            abs_tolerance=tol_abs, rel_tolerance=tol_rel,
        )
    elif (
        predicted and gold
        and is_scalar_numeric_result(predicted)
        and is_scalar_numeric_result(gold)
    ):
        metrics["typed_metrics"] = compare_scalar_results(
            predicted, gold, abs_tolerance=tol_abs, rel_tolerance=tol_rel,
        )
    else:
        metrics["typed_metrics"] = compare_rowset_results(predicted, gold)
    return metrics


# ---------------------------------------------------------------------------
# Per-stem summary
# ---------------------------------------------------------------------------

def build_stem_summary(
    results: list[dict[str, Any]],
    stem: str,
    query_type: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    timeouts    = sum(1 for r in results if (r.get("generation_error") or "").startswith("generation_timeout"))
    sql_ok      = sum(1 for r in results if r.get("predicted_sql"))
    exec_ok     = sum(1 for r in results if not (r.get("predicted_exec") or {}).get("error"))
    gen_times   = [r["timing"]["generation_seconds"] for r in results if r.get("timing")]
    exec_times  = [r["timing"]["execution_seconds"]  for r in results if r.get("timing")]
    total_times = [r["timing"]["total_seconds"]       for r in results if r.get("timing")]
    p_tok       = [r["usage"]["prompt_tokens"]        for r in results if (r.get("usage") or {}).get("prompt_tokens")     is not None]
    c_tok       = [r["usage"]["completion_tokens"]    for r in results if (r.get("usage") or {}).get("completion_tokens") is not None]
    t_tok       = [r["usage"]["total_tokens"]         for r in results if (r.get("usage") or {}).get("total_tokens")      is not None]

    summary: dict[str, Any] = {
        "stem":                      stem,
        "query_type":                query_type,
        "model":                     args.model,
        "provider":                  args.provider,
        "question_count":            len(results),
        "generation_timeouts":       timeouts,
        "successful_sql_generation": sql_ok,
        "successful_sql_execution":  exec_ok,
        "timing": {
            "total_wall_seconds":      round(sum(total_times),             3) if total_times else 0.0,
            "mean_generation_seconds": round(statistics.mean(gen_times),   3) if gen_times   else None,
            "mean_execution_seconds":  round(statistics.mean(exec_times),  3) if exec_times  else None,
            "mean_total_seconds":      round(statistics.mean(total_times), 3) if total_times else None,
            "max_total_seconds":       round(max(total_times),             3) if total_times else None,
        },
        "tokens": {
            "total_prompt_tokens":     sum(p_tok) if p_tok else None,
            "total_completion_tokens": sum(c_tok) if c_tok else None,
            "total_tokens":            sum(t_tok) if t_tok else None,
            "mean_prompt_tokens":      round(statistics.mean(p_tok), 1) if p_tok else None,
            "mean_completion_tokens":  round(statistics.mean(c_tok), 1) if c_tok else None,
            "mean_total_tokens":       round(statistics.mean(t_tok), 1) if t_tok else None,
        },
    }

    if not args.generate_only:
        comparisons  = [r.get("comparison", {}) for r in results if r.get("comparison")]
        exact        = sum(1 for c in comparisons if c.get("exact_match"))
        typed_list   = [c["typed_metrics"] for c in comparisons if "typed_metrics" in c]
        numeric_tol  = [t for t in typed_list if t["metric_family"] in ("scalar_numeric", "primary_numeric")]
        text_exact   = [t for t in typed_list if t["metric_family"] == "primary_text"]
        rowset       = [t for t in typed_list if t["metric_family"] == "rowset"]
        summary["evaluation"] = {
            "exact_matches":              exact,
            "numeric_tolerance_accuracy": round(
                sum(1 for t in numeric_tol if t["within_tolerance"]) / len(numeric_tol), 4
            ) if numeric_tol else None,
            "mean_absolute_error":        round(
                statistics.mean(
                    t["absolute_error"] for t in numeric_tol
                    if t.get("absolute_error") is not None
                ), 4
            ) if any(t.get("absolute_error") is not None for t in numeric_tol) else None,
            "text_exact_accuracy":        round(
                sum(1 for t in text_exact if t["exact_match"]) / len(text_exact), 4
            ) if text_exact else None,
            "mean_rowset_f1":             round(
                statistics.mean(t["f1"] for t in rowset), 4
            ) if rowset else None,
        }
    return summary


# ---------------------------------------------------------------------------
# Process one question (worker function — each call gets its own DB conn)
# ---------------------------------------------------------------------------

def process_question(
    question: dict[str, Any],
    idx: int,
    total: int,
    stem: str,
    stem_dir: Path,
    system_prompt: str,
    db_params: dict[str, Any],
    args: argparse.Namespace,
    client: OpenAI | None,
    gemini_api_key: str | None,
    query_type: str,
) -> dict[str, Any]:
    """Process a single question. Opens its own DB connection (thread-safe)."""
    qid    = question["id"]
    cached = load_question_result(stem_dir, qid)
    if cached is not None and (args.generate_only or "comparison" in cached):
        print(f"  [skip] {qid}")
        return cached

    tag             = f"[{query_type}/{stem} {idx}/{total}]"
    exec_timeout_ms = args.execution_timeout * 1000

    # ---- generation ----
    gen_start    = time.perf_counter()
    model_result = invoke_model_with_timeout(
        timeout_seconds=args.generation_timeout,
        provider=args.provider,
        model=args.model,
        system_prompt=system_prompt,
        user_prompt=build_user_prompt(question),
        openai_client=client,
        gemini_api_key=gemini_api_key,
    )
    gen_elapsed      = round(time.perf_counter() - gen_start, 3)
    generation_error = model_result.pop("generation_error", "")
    usage            = model_result.get("usage", {})

    # ---- execution (own connection per worker) ----
    conn = psycopg.connect(**db_params)
    try:
        exec_start = time.perf_counter()
        if generation_error.startswith("generation_timeout"):
            sql_blocks     = []
            predicted_sql  = ""
            predicted_exec = {"output": [], "error": "skipped_generation_timeout"}
        else:
            sql_blocks = extract_sql_blocks(model_result.get("text", ""))
            predicted_sql, predicted_exec = execute_sql_candidates(
                conn, sql_blocks, timeout_ms=exec_timeout_ms,
            )
        exec_elapsed  = round(time.perf_counter() - exec_start, 3)

        gold_sql  = question.get("sql", "")
        gold_exec = (
            run_sql(conn, gold_sql, timeout_ms=exec_timeout_ms)
            if gold_sql
            else {"output": [], "error": "No gold SQL."}
        )
    finally:
        conn.close()

    total_elapsed = round(gen_elapsed + exec_elapsed, 3)

    if generation_error:
        print(f"  {tag} ERROR — {generation_error}")
    else:
        print(
            f"  {tag} gen={gen_elapsed:.1f}s  exec={exec_elapsed:.1f}s  "
            f"total={total_elapsed:.1f}s  "
            f"tokens={usage.get('total_tokens', 'n/a')}  "
            f"sql={'yes' if predicted_sql else 'no'}  "
            f"exec={'ok' if not predicted_exec['error'] else 'FAIL'}"
        )

    record: dict[str, Any] = {
        "id":                       qid,
        "source_file":              question["source_file"],
        "source_stem":              stem,
        "query_type":               query_type,
        "question":                 question["question"],
        "answer_type":              question.get("answer_type"),
        "gold_answers":             question.get("answers", []),
        "model_output":             model_result.get("text", ""),
        "generation_error":         generation_error,
        "predicted_sql_candidates": sql_blocks,
        "predicted_sql":            predicted_sql,
        "predicted_exec":           predicted_exec,
        "gold_sql":                 gold_sql,
        "gold_exec":                gold_exec,
        "usage": {
            "prompt_tokens":     usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens":      usage.get("total_tokens"),
        },
        "timing": {
            "generation_seconds": gen_elapsed,
            "execution_seconds":  exec_elapsed,
            "total_seconds":      total_elapsed,
        },
    }

    if not args.generate_only:
        record["comparison"] = compare_results(
            question,
            predicted_exec["output"],
            gold_exec["output"],
            answer_type=question.get("answer_type"),
            abs_tolerance=args.numeric_abs_tolerance,
            rel_tolerance=args.numeric_rel_tolerance,
        )

    save_question_result(stem_dir, record)
    return record


# ---------------------------------------------------------------------------
# Process one stem (one JSONL file)
# ---------------------------------------------------------------------------

def process_stem(
    questions: list[dict[str, Any]],
    stem: str,
    stem_dir: Path,
    system_prompt: str,
    db_params: dict[str, Any],
    args: argparse.Namespace,
    client: OpenAI | None,
    gemini_api_key: str | None,
    query_type: str,
) -> list[dict[str, Any]]:
    stem_dir.mkdir(parents=True, exist_ok=True)
    total   = len(questions)
    workers = max(1, args.workers)

    common = dict(
        stem=stem,
        stem_dir=stem_dir,
        system_prompt=system_prompt,
        db_params=db_params,
        args=args,
        client=client,
        gemini_api_key=gemini_api_key,
        query_type=query_type,
        total=total,
    )

    if workers == 1:
        # sequential — predictable log order
        results = [
            process_question(question=q, idx=i, **common)
            for i, q in enumerate(questions, start=1)
        ]
    else:
        print(f"  parallel workers={workers}")
        results_by_id: dict[str, dict[str, Any]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(process_question, question=q, idx=i, **common): q["id"]
                for i, q in enumerate(questions, start=1)
            }
            for future in concurrent.futures.as_completed(futures):
                qid = futures[future]
                try:
                    results_by_id[qid] = future.result()
                except Exception as exc:  # noqa: BLE001
                    print(f"  [ERROR] {qid}: {exc}")
        # preserve original question order
        results = [results_by_id[q["id"]] for q in questions if q["id"] in results_by_id]

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args          = parse_args()
    benchmark_dir = Path(args.benchmark_dir)
    cache_dir     = Path(args.cache_dir)

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    gemini_api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if args.provider == "openai" and not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required.")
    if args.provider == "gemini" and not gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is required.")

    client_kwargs = {"api_key": openai_api_key}
    if args.openai_base_url:
        client_kwargs["base_url"] = args.openai_base_url
    client        = OpenAI(**client_kwargs) if args.provider == "openai" else None
    system_prompt = DEFAULT_SYSTEM_PROMPT.format(dem_table=args.dem_table)

    # db_params is passed to each worker so it can open its own connection
    db_params = dict(
        host=args.db_host,
        dbname=args.db_name,
        user=args.db_user,
        password=args.db_password,
        port=args.db_port,
    )

    # verify connectivity once before starting
    test_conn = psycopg.connect(**db_params)
    test_conn.close()

    global_start  = time.perf_counter()
    all_summaries: list[dict[str, Any]] = []

    for query_type in args.query_types:
        input_dir = benchmark_dir / query_type
        if not input_dir.exists():
            print(f"[skip] {input_dir} — not found")
            continue

        # baselines/cache/{provider}_{query_type}/
        type_cache_dir = cache_dir / f"{args.provider}_{query_type}"
        type_cache_dir.mkdir(parents=True, exist_ok=True)
        (type_cache_dir / "prompt.txt").write_text(system_prompt, encoding="utf-8")

        questions = load_questions(input_dir, args.samples_per_file)
        if not questions:
            print(f"[skip] no JSONL files in {input_dir}")
            continue

        print(f"\n{'='*64}")
        print(f"  query_type : {query_type}")
        print(f"  questions  : {len(questions)}")
        print(f"  workers    : {args.workers}")
        print(f"  output     : {type_cache_dir}")
        print(f"{'='*64}")

        # group by JSONL stem
        stems: dict[str, list[dict[str, Any]]] = {}
        for q in questions:
            stems.setdefault(q["source_stem"], []).append(q)

        type_summaries: list[dict[str, Any]] = []
        for stem, stem_questions in stems.items():
            stem_dir = type_cache_dir / stem
            print(f"\n--- {stem}  ({len(stem_questions)} questions) ---")

            stem_results = process_stem(
                questions=stem_questions,
                stem=stem,
                stem_dir=stem_dir,
                system_prompt=system_prompt,
                db_params=db_params,
                args=args,
                client=client,
                gemini_api_key=gemini_api_key,
                query_type=query_type,
            )

            stem_summary = build_stem_summary(stem_results, stem, query_type, args)
            (stem_dir / "summary.json").write_text(
                json.dumps(stem_summary, indent=2), encoding="utf-8"
            )
            type_summaries.append(stem_summary)
            all_summaries.append(stem_summary)
            print(f"  -> {stem_dir / 'summary.json'}")

        # aggregate across stems for this query type
        run_summary = {
            "query_type":      query_type,
            "provider":        args.provider,
            "model":           args.model,
            "workers":         args.workers,
            "stems_processed": len(type_summaries),
            "total_questions": sum(s["question_count"]           for s in type_summaries),
            "total_timeouts":  sum(s["generation_timeouts"]      for s in type_summaries),
            "total_sql_ok":    sum(s["successful_sql_generation"] for s in type_summaries),
            "total_exec_ok":   sum(s["successful_sql_execution"]  for s in type_summaries),
            "total_tokens":    sum((s["tokens"]["total_tokens"] or 0) for s in type_summaries),
            "total_wall_seconds": round(
                sum(s["timing"]["total_wall_seconds"] for s in type_summaries), 1
            ),
            "stems": type_summaries,
        }
        (type_cache_dir / "run_summary.json").write_text(
            json.dumps(run_summary, indent=2), encoding="utf-8"
        )
        print(f"\nrun_summary -> {type_cache_dir / 'run_summary.json'}")

    elapsed = round(time.perf_counter() - global_start, 1)
    print(f"\nFinished in {elapsed}s — {len(all_summaries)} stems across {len(args.query_types)} query type(s)")


if __name__ == "__main__":
    main()
