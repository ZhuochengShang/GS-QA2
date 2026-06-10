#!/usr/bin/env python3
"""evaluate_extended.py — unified evaluation for raster, raster-vector,
extended, and vector-only query results.

Reads per-question JSON files produced by run_raster_text2sql.py from:
  baselines/cache/{provider}_{query_type}/{stem}/{stem}-N.json

Reads vector-only CSV results produced by baselines.py from:
  baselines/cache/  (or --vector-csv path)

Applies per-query-type tolerances from Table 6 / Table 7:

  Elevation (abs ≤ 10 m):
    elevation+poi, elevation+region max/min/mean/range,
    elevation difference + two POIs

  Elevation coverage (abs ≤ 5 pp):
    elevation+coverage

  Exact Match (correct label):
    elevation threshold / comparison
    slope / elevation comparison

  Slope & Aspect (abs ≤ 5°):
    slope+poi/route, slope+area max/min/average, aspect+poi

  Ruggedness (rel ≤ 5%):
    ruggedness+poi

  Token F1 (≥ 0.8):
    name+elevation/slope condition
    lowest/highest/steepest POI

  Distance Error (≤ 5 m):
    location+elevation condition

  Relative Error (≤ 5%):
    area/length/distance, nearest high/low terrain, count+elevation

Vector-only (from baselines.py CSVs):
  Entity name → Token F1 ≥ 0.8
  Location    → Distance Error ≤ 5 m
  Area/length/distance/count → Relative Error ≤ 5%
  Direction   → Angular Error ≤ 5°

Outputs:
  evaluation_summary.json   — machine-readable full results
  evaluation_report.txt     — human-readable table
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Metric routing table  (source_stem substring → metric config)
# ---------------------------------------------------------------------------

# Each entry: (metric_type, tolerance_value, tolerance_unit)
# Checked in order — first match wins.
STEM_METRIC_MAP: list[tuple[str, str, float, str]] = [
    # --- elevation ---
    ("elevation_coverage",          "absolute",  5.0,   "pp"),
    ("elevation_threshold",         "exact",     0.0,   "label"),
    ("elevation_compare",           "exact",     0.0,   "label"),
    ("elevation_diff",              "absolute",  10.0,  "m"),
    ("elevation",                   "absolute",  10.0,  "m"),
    # --- slope / aspect / ruggedness ---
    ("slope_compare",               "exact",     0.0,   "label"),
    ("slope_elevation_compare",     "exact",     0.0,   "label"),
    ("aspect",                      "angular",   5.0,   "deg"),
    ("ruggedness",                  "relative",  0.05,  "%"),
    ("slope",                       "absolute",  5.0,   "deg"),
    # --- name / entity ---
    ("name",                        "token_f1",  0.8,   "f1"),
    ("intersects_area_max_name",    "token_f1",  0.8,   "f1"),
    ("range_name",                  "token_f1",  0.8,   "f1"),
    ("nearest_high",                "relative",  0.05,  "%"),
    ("nearest_low",                 "relative",  0.05,  "%"),
    # --- numeric spatial ---
    ("count",                       "relative",  0.05,  "%"),
    ("area",                        "relative",  0.05,  "%"),
    ("length",                      "relative",  0.05,  "%"),
    ("distance",                    "relative",  0.05,  "%"),
    # --- location ---
    ("location",                    "distance",  5.0,   "m"),
]

# Fallback if no stem matches — use answer_type
ANSWER_TYPE_MAP: list[tuple[str, str, float, str]] = [
    ("yes-no",              "exact",     0.0,   "label"),
    ("name",                "token_f1",  0.8,   "f1"),
    ("coverage",            "absolute",  5.0,   "pp"),
    ("mean elevation",      "absolute",  10.0,  "m"),
    ("elevation",           "absolute",  10.0,  "m"),
    ("aspect",              "angular",   5.0,   "deg"),
    ("slope",               "absolute",  5.0,   "deg"),
    ("ruggedness",          "relative",  0.05,  "%"),
    ("distance",            "relative",  0.05,  "%"),
    ("area",                "relative",  0.05,  "%"),
    ("count",               "relative",  0.05,  "%"),
]


def get_metric_config(
    source_stem: str,
    answer_type: str | None,
) -> tuple[str, float, str]:
    """Return (metric_type, tolerance, unit) for a question."""
    stem_lower = (source_stem or "").lower()
    for key, metric, tol, unit in STEM_METRIC_MAP:
        if key in stem_lower:
            return metric, tol, unit
    at_lower = (answer_type or "").lower()
    for key, metric, tol, unit in ANSWER_TYPE_MAP:
        if key in at_lower:
            return metric, tol, unit
    return "relative", 0.05, "%"   # safe default


# ---------------------------------------------------------------------------
# Text helpers (mirrors evaluate.py)
# ---------------------------------------------------------------------------

def _normalize_tokens(s: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", str(s).lower()))


def token_f1(pred: str, gold: str) -> tuple[float, float, float]:
    pt = Counter(_normalize_tokens(pred))
    gt = Counter(_normalize_tokens(gold))
    if not pt or not gt:
        return 0.0, 0.0, 0.0
    common = pt & gt
    if not common:
        return 0.0, 0.0, 0.0
    common_count = sum(common.values())
    p   = common_count / sum(pt.values())
    r   = common_count / sum(gt.values())
    f1  = 2 * p * r / (p + r)
    return p, r, f1


# ---------------------------------------------------------------------------
# Field extraction from SQL output rows
# ---------------------------------------------------------------------------

NUMERIC_PREFERRED = (
    "elevation", "elevation_m", "elevation_difference_m",
    "mean_elevation", "max_elevation", "min_elevation", "elevation_range",
    "slope_degrees", "slope", "aspect_degrees", "aspect",
    "coverage", "coverage_percent", "percent", "percentage",
    "distance_m", "area", "length", "count", "ruggedness",
    "avg_slope", "min_slope", "max_slope",
)

TEXT_PREFERRED = (
    "answer", "higher_poi", "steeper_poi", "answer_name",
    "name", "poi_name", "road_name", "park_name", "lake_name", "region_name",
)


def _pick_field(rows: list[dict], preferred: tuple) -> tuple[str | None, Any]:
    if not rows:
        return None, None
    row = rows[0]
    for f in preferred:
        if f in row:
            return f, row[f]
    # fallback: first matching type
    return None, None


def extract_numeric(rows: list[dict]) -> float | None:
    _, val = _pick_field(rows, NUMERIC_PREFERRED)
    if val is None:
        # fallback: first numeric value in first row
        if rows and isinstance(rows[0], dict):
            for v in rows[0].values():
                if isinstance(v, (int, float)):
                    return float(v)
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def extract_text(rows: list[dict]) -> str | None:
    _, val = _pick_field(rows, TEXT_PREFERRED)
    if val is not None:
        return str(val)
    # fallback: first string value
    if rows and isinstance(rows[0], dict):
        for v in rows[0].values():
            if isinstance(v, str) and v.strip():
                return v.strip()
    return None


# ---------------------------------------------------------------------------
# Ground-truth-aware component evaluation
# ---------------------------------------------------------------------------

NAME_FIELDS = (
    "answer", "answer_name", "higher_poi", "steeper_poi",
    "name", "poi_name", "road_name", "park_name", "lake_name", "region_name",
)

COMPONENT_SPECS: dict[str, list[tuple[str, str, tuple[str, ...], float, str]]] = {
    # raster-only
    "aspect+poi": [("aspect", "angular", ("aspect_degrees", "aspect"), 5.0, "deg")],
    "avg_slope+area": [("slope", "absolute", ("slope_degrees", "avg_slope", "slope"), 5.0, "deg")],
    "elevation+coverage": [("coverage", "absolute", ("coverage_percent", "coverage", "percent", "percentage"), 5.0, "pp")],
    "elevation+poi": [("elevation", "absolute", ("elevation", "elevation_m"), 10.0, "m")],
    "elevation_compare+two_pois": [("name", "token_f1", ("higher_poi", "higher_location", "higher_place", "answer_name", "name"), 0.8, "f1")],
    "elevation_diff+two_pois": [("elevation_difference", "absolute", ("elevation_difference_m", "elevation_difference", "difference"), 10.0, "m")],
    "elevation_max+region": [("elevation", "absolute", ("max_elevation", "elevation", "max"), 10.0, "m")],
    "elevation_mean+region": [("elevation", "absolute", ("mean_elevation", "mean", "elevation"), 10.0, "m")],
    "elevation_min+region": [("elevation", "absolute", ("min_elevation", "elevation", "min"), 10.0, "m")],
    "elevation_range+region": [("elevation", "absolute", ("elevation_range", "range", "elevation"), 10.0, "m")],
    "elevation_threshold+poi": [("label", "exact", ("answer",), 0.0, "label")],
    "max_slope+area": [("slope", "absolute", ("max_slope", "slope_degrees", "slope"), 5.0, "deg")],
    "min_slope+area": [("slope", "absolute", ("min_slope", "slope_degrees", "slope"), 5.0, "deg")],
    "nearest_high_terrain+poi": [("distance", "relative", ("distance_m", "distance"), 0.05, "%")],
    "nearest_low_terrain+poi": [("distance", "relative", ("distance_m", "distance"), 0.05, "%")],
    "ruggedness+poi": [("ruggedness", "relative", ("tri_value", "ruggedness", "ruggedness_index"), 0.05, "%")],
    "slope+compare_route": [("name", "token_f1", ("answer_name", "answer", "name"), 0.8, "f1")],
    "slope+poi": [("slope", "absolute", ("slope_degrees", "slope"), 5.0, "deg")],
    "slope+route": [("slope", "absolute", ("slope_degrees", "avg_slope", "slope"), 5.0, "deg")],
    # raster-vector: terrain values are filters unless explicitly requested
    "elevation_average+name": [("name", "token_f1_set", NAME_FIELDS, 0.8, "f1")],
    "elevation_compare_categories+name": [("name", "token_f1", ("answer_name", "answer", "name"), 0.8, "f1")],
    "elevation_count_threshold+count": [("count", "relative", ("count",), 0.05, "%")],
    "elevation_lowest_n_poi+name": [("name", "token_f1_set", NAME_FIELDS, 0.8, "f1")],
    "elevation_proximity_zone+name": [("name", "token_f1_set", NAME_FIELDS, 0.8, "f1")],
    "slope_condition+name": [("name", "token_f1_set", NAME_FIELDS, 0.8, "f1")],
    "slope_steepest_n_poi+name": [("name", "token_f1_set", NAME_FIELDS, 0.8, "f1")],
    # extended compound outputs
    "intersects_area_max+name+max_elevation": [
        ("name", "token_f1", NAME_FIELDS, 0.8, "f1"),
        ("elevation", "absolute", ("max_elevation", "max_elev", "elevation"), 10.0, "m"),
    ],
    "intersects_area_total+area+avg_slope": [
        ("area", "relative", ("area", "computed_area"), 0.05, "%"),
        ("slope", "absolute", ("avg_slope", "slope"), 5.0, "deg"),
    ],
    "intersects_length_max+name+slope": [
        ("name", "token_f1", NAME_FIELDS, 0.8, "f1"),
        ("slope", "absolute", ("slope_degrees", "slope_deg", "slope"), 5.0, "deg"),
    ],
    "knn+distance+slope": [
        ("name", "token_f1", NAME_FIELDS, 0.8, "f1"),
        ("distance", "relative", ("dist_m", "distance_m", "distance"), 0.05, "%"),
        ("slope", "absolute", ("slope_deg", "slope_degrees", "slope"), 5.0, "deg"),
    ],
    "range+count+elevation_condition": [("count", "relative", ("count",), 0.05, "%")],
    "range+distance+elevation": [
        ("distance", "relative", ("distance", "distance_m", "dist_m"), 0.05, "%"),
        ("elevation", "absolute", ("elevation", "elevation_m"), 10.0, "m"),
    ],
    "range+loc+elevation_condition": [("location", "location", ("geometry",), 5.0, "m")],
    "range+name+elevation_condition": [("name", "token_f1_set", NAME_FIELDS, 0.8, "f1")],
}


def _field_value(
    row: dict[str, Any],
    aliases: tuple[str, ...],
    kind: str | None = None,
    allow_singleton_fallback: bool = True,
) -> Any:
    def matches(value: Any) -> bool:
        if value in (None, ""):
            return False
        if kind == "text":
            return isinstance(value, str)
        if kind == "number":
            return _as_number(value) is not None
        return True

    for alias in aliases:
        if alias in row and matches(row[alias]):
            return row[alias]
    for alias in aliases:
        for key, value in row.items():
            if alias in str(key).lower() and matches(value):
                return value
    if allow_singleton_fallback:
        values = [value for value in row.values() if matches(value)]
        if len(values) == 1:
            return values[0]
    return None


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(match.group(0)) if match else None


def _normalize_text(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())


def _point_from_wkt(value: Any) -> tuple[float, float] | None:
    match = re.search(
        r"POINT\s*(?:Z\s*)?\(\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)",
        str(value), re.I,
    )
    if match:
        return float(match.group(1)), float(match.group(2))
    try:
        from shapely import wkb
        geom = wkb.loads(bytes.fromhex(str(value)))
        if geom.geom_type == "Point":
            return float(geom.x), float(geom.y)
    except (ImportError, TypeError, ValueError):
        pass
    return None


def _haversine_m(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    lon1, lat1 = point_a
    lon2, lat2 = point_b
    radius = 6371008.8
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _component_result(
    pred_row: dict[str, Any],
    gold_row: dict[str, Any],
    spec: tuple[str, str, tuple[str, ...], float, str],
) -> dict[str, Any]:
    name, metric, aliases, tolerance, unit = spec
    kind = "text" if metric in ("exact", "token_f1", "location") else "number"
    allow_singleton_fallback = metric not in ("token_f1",)
    pred = _field_value(pred_row, aliases, kind=kind, allow_singleton_fallback=allow_singleton_fallback)
    gold = _field_value(gold_row, aliases, kind=kind, allow_singleton_fallback=allow_singleton_fallback)
    result = {"component": name, "metric": metric, "tolerance": tolerance, "unit": unit}
    if metric in ("exact", "token_f1"):
        if pred is None or gold is None:
            return {**result, "passed": False, "score": 0.0, "reason": "missing_value"}
        if metric == "exact":
            score = float(_normalize_text(pred) == _normalize_text(gold))
            return {**result, "passed": bool(score), "score": score, "predicted": pred, "gold": gold}
        _, _, f1 = token_f1(str(pred), str(gold))
        return {**result, "passed": f1 >= tolerance, "score": round(f1, 4), "predicted": pred, "gold": gold}
    if metric == "location":
        pred_point = _point_from_wkt(pred)
        gold_point = _point_from_wkt(gold)
        if pred_point is None or gold_point is None:
            return {**result, "passed": False, "score": None, "reason": "missing_geometry"}
        error = _haversine_m(pred_point, gold_point)
        return {**result, "passed": error <= tolerance, "score": round(error, 4), "predicted": pred, "gold": gold}
    pred_num, gold_num = _as_number(pred), _as_number(gold)
    if pred_num is None or gold_num is None:
        return {**result, "passed": False, "score": None, "reason": "non_numeric_output"}
    if metric == "absolute":
        error = abs(pred_num - gold_num)
    elif metric == "angular":
        delta = abs(pred_num - gold_num) % 360
        error = min(delta, 360 - delta)
    else:
        error = 0.0 if pred_num == gold_num == 0 else (
            float("inf") if gold_num == 0 else abs(pred_num - gold_num) / abs(gold_num)
        )
    return {
        **result, "passed": error <= tolerance,
        "score": round(error, 4) if math.isfinite(error) else None,
        "predicted": pred_num, "gold": gold_num,
    }


def _text_set_result(
    predicted: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    spec: tuple[str, str, tuple[str, ...], float, str],
) -> dict[str, Any]:
    name, metric, aliases, tolerance, unit = spec
    pred_text = "\n".join(
        str(v) for row in predicted
        if (v := _field_value(row, aliases, kind="text", allow_singleton_fallback=False)) is not None
    )
    gold_text = "\n".join(
        str(v) for row in gold
        if (v := _field_value(row, aliases, kind="text", allow_singleton_fallback=False)) is not None
    )
    _, _, f1 = token_f1(pred_text, gold_text)
    return {
        "component": name, "metric": metric, "tolerance": tolerance, "unit": unit,
        "passed": bool(pred_text and gold_text and f1 >= tolerance), "score": round(f1, 4),
        "predicted": pred_text, "gold": gold_text,
    }


def _evaluate_components(
    source_stem: str,
    predicted: list[dict[str, Any]],
    gold: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    specs = COMPONENT_SPECS.get(source_stem)
    if specs is None:
        return None
    set_specs = [spec for spec in specs if spec[1] == "token_f1_set"]
    pair_specs = [spec for spec in specs if spec[1] != "token_f1_set"]
    components = [_text_set_result(predicted, gold, spec) for spec in set_specs]
    if pair_specs:
        candidates = [
            [_component_result(pred_row, gold_row, spec) for spec in pair_specs]
            for pred_row in predicted for gold_row in gold
        ]
        if not candidates:
            candidates = [[
                {"component": name, "metric": metric, "tolerance": tolerance, "unit": unit,
                 "passed": False, "score": None, "reason": "empty_output"}
                for name, metric, _, tolerance, unit in pair_specs
            ]]
        components.extend(max(candidates, key=lambda items: (sum(bool(item["passed"]) for item in items), -sum(item["score"] or 0 for item in items))))
    return components


# ---------------------------------------------------------------------------
# Per-question evaluation
# ---------------------------------------------------------------------------

def evaluate_question(record: dict[str, Any]) -> dict[str, Any]:
    source_stem  = record.get("source_stem", "")
    answer_type  = record.get("answer_type", "")
    pred_rows    = (record.get("predicted_exec") or {}).get("output", [])
    gold_rows    = (record.get("gold_exec")      or {}).get("output", []) or record.get("gold_answers", [])
    pred_error   = (record.get("predicted_exec") or {}).get("error", "")
    generation_error = record.get("generation_error", "")

    metric, tolerance, unit = get_metric_config(source_stem, answer_type)

    result: dict[str, Any] = {
        "id":           record.get("id"),
        "query_type":   record.get("query_type", "unknown"),
        "source_stem":  source_stem,
        "answer_type":  answer_type,
        "metric":       metric,
        "tolerance":    tolerance,
        "unit":         unit,
        "sql_generated": bool(record.get("predicted_sql")),
        "sql_executed":  not bool(pred_error),
        "generation_error": generation_error,
        "correct":      False,
        "score":        0.0,
        "detail":       {},
    }

    components = _evaluate_components(source_stem, pred_rows, gold_rows)
    if components is not None:
        result["metric"] = "compound" if len(components) > 1 else components[0]["metric"]
        result["tolerance"] = None if len(components) > 1 else components[0]["tolerance"]
        result["unit"] = None if len(components) > 1 else components[0]["unit"]
        result["correct"] = bool(components) and all(component["passed"] for component in components)
        scores = [component["score"] for component in components if component.get("score") is not None]
        result["score"] = round(statistics.mean(scores), 4) if scores else 0.0
        result["detail"] = {"components": components}
        return result

    # no output → wrong
    if not pred_rows or not gold_rows:
        result["detail"]["reason"] = (
            "generation_timeout" if "timeout" in generation_error
            else "sql_error" if pred_error
            else "empty_output"
        )
        return result

    if metric == "exact":
        pred_val = extract_text(pred_rows) or ""
        gold_val = extract_text(gold_rows) or ""
        # also accept numeric exact for yes/no
        if pred_val == "" and pred_rows:
            pred_val = str(list(pred_rows[0].values())[0]).strip().lower()
        if gold_val == "" and gold_rows:
            gold_val = str(list(gold_rows[0].values())[0]).strip().lower()
        match    = pred_val.strip().lower() == gold_val.strip().lower()
        result["correct"] = match
        result["score"]   = float(match)
        result["detail"]  = {"predicted": pred_val, "gold": gold_val}

    elif metric == "token_f1":
        pred_val = extract_text(pred_rows) or ""
        gold_val = extract_text(gold_rows) or ""
        _, _, f1 = token_f1(pred_val, gold_val)
        result["correct"] = f1 >= tolerance
        result["score"]   = round(f1, 4)
        result["detail"]  = {"predicted": pred_val, "gold": gold_val, "f1": round(f1, 4)}

    elif metric == "absolute":
        pred_val = extract_numeric(pred_rows)
        gold_val = extract_numeric(gold_rows)
        if pred_val is None or gold_val is None:
            result["detail"]["reason"] = "non_numeric_output"
        else:
            abs_err = abs(pred_val - gold_val)
            result["correct"] = abs_err <= tolerance
            result["score"]   = round(abs_err, 4)
            result["detail"]  = {
                "predicted":      round(pred_val, 4),
                "gold":           round(gold_val, 4),
                "absolute_error": round(abs_err, 4),
            }

    elif metric == "angular":
        pred_val = extract_numeric(pred_rows)
        gold_val = extract_numeric(gold_rows)
        if pred_val is None or gold_val is None:
            result["detail"]["reason"] = "non_numeric_output"
        else:
            delta = abs(pred_val - gold_val) % 360
            err = min(delta, 360 - delta)
            result["correct"] = err <= tolerance
            result["score"]   = round(err, 4)
            result["detail"]  = {
                "predicted":    round(pred_val, 4),
                "gold":         round(gold_val, 4),
                "angular_error": round(err, 4),
            }

    elif metric == "relative":
        pred_val = extract_numeric(pred_rows)
        gold_val = extract_numeric(gold_rows)
        if pred_val is None or gold_val is None:
            result["detail"]["reason"] = "non_numeric_output"
        elif gold_val == 0:
            rel_err = 0.0 if pred_val == 0 else float("inf")
            result["correct"] = pred_val == 0
            result["score"] = 0.0 if pred_val == 0 else None
            result["detail"] = {
                "predicted": round(pred_val, 4),
                "gold": round(gold_val, 4),
                "relative_error": result["score"],
            }
        else:
            rel_err = abs(pred_val - gold_val) / abs(gold_val)
            result["correct"] = rel_err <= tolerance
            result["score"]   = round(rel_err, 4)
            result["detail"]  = {
                "predicted":      round(pred_val, 4),
                "gold":           round(gold_val, 4),
                "relative_error": round(rel_err, 4),
            }

    elif metric == "distance":
        # expects output with lon/lat fields or a numeric distance_m
        pred_val = extract_numeric(pred_rows)
        gold_val = extract_numeric(gold_rows)
        if pred_val is None or gold_val is None:
            result["detail"]["reason"] = "non_numeric_output"
        else:
            abs_err = abs(pred_val - gold_val)
            result["correct"] = abs_err <= tolerance
            result["score"]   = round(abs_err, 4)
            result["detail"]  = {
                "predicted":    round(pred_val, 4),
                "gold":         round(gold_val, 4),
                "distance_error": round(abs_err, 4),
            }

    return result


# ---------------------------------------------------------------------------
# Load raster/raster-vector/extended results from cache
# ---------------------------------------------------------------------------

def load_cache_results(
    cache_dir: Path,
    provider: str,
    query_types: list[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for qt in query_types:
        type_dir = cache_dir / f"{provider}_{qt}"
        if not type_dir.exists():
            print(f"[skip] {type_dir} not found")
            continue
        for stem_dir in sorted(type_dir.iterdir()):
            if not stem_dir.is_dir():
                continue
            for f in sorted(stem_dir.glob("*.json")):
                if f.name in ("summary.json", "run_summary.json"):
                    continue
                try:
                    rec = json.loads(f.read_text(encoding="utf-8"))
                    if rec.get("query_type") in (None, "", "unknown"):
                        rec["query_type"] = qt
                    records.append(rec)
                except Exception as exc:
                    print(f"  [warn] could not read {f}: {exc}")
    return records


# ---------------------------------------------------------------------------
# Load vector-only CSV results from baselines.py
# ---------------------------------------------------------------------------

def load_vector_csv(csv_path: Path) -> list[dict[str, Any]]:
    """Read the parsed_eval CSV produced by baselines.py save_eval()."""
    import csv
    records = []
    if not csv_path.exists():
        return records
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            records.append(row)
    return records


def evaluate_vector_row(row: dict[str, str]) -> dict[str, Any]:
    """Evaluate one row from baselines.py parsed_eval CSV."""
    qtype = str(row.get("type", "")).lower()

    if "name" in qtype or "multi_source" in qtype:
        metric, tol, unit = "token_f1", 0.8, "f1"
        try:
            f1      = float(row.get("F1", 0))
            correct = f1 >= tol
            score   = round(f1, 4)
        except (ValueError, TypeError):
            correct, score = False, 0.0

    elif "angle" in qtype:
        metric, tol, unit = "angular", 5.0 / 180.0, "normalized"
        try:
            # baselines.py stores GS-QA normalized circular angle error.
            err     = float(row.get("angle_error", 1.0))
            correct = err <= tol
            score   = round(err, 4)
        except (ValueError, TypeError):
            correct, score = False, 1.0

    elif "loc" in qtype:
        metric, tol, unit = "distance", 5.0 / 500000.0, "normalized"
        try:
            # baselines.py stores location error normalized by 500 km.
            err     = float(row.get("distance_error", 1.0))
            correct = err <= tol
            score   = round(err, 4)
        except (ValueError, TypeError):
            correct, score = False, 1.0

    else:  # area, length, count, distance
        metric, tol, unit = "relative", 0.05, "%"
        try:
            rel     = float(row.get("relative_error", 1.0))
            correct = rel <= tol
            score   = round(rel, 4)
        except (ValueError, TypeError):
            correct, score = False, 1.0

    return {
        "id":          row.get("id", ""),
        "source_stem": "vector_only",
        "query_type":  "vector_only",
        "answer_type": row.get("type", ""),
        "metric":      metric,
        "tolerance":   tol,
        "unit":        unit,
        "sql_generated": True,
        "sql_executed":  row.get("attempted", "False") == "True",
        "generation_error": "",
        "correct":     correct,
        "score":       score,
        "detail":      dict(row),
    }


# ---------------------------------------------------------------------------
# Summary builders
# ---------------------------------------------------------------------------

def build_stem_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    total   = len(results)
    correct = sum(1 for r in results if r["correct"])
    scores  = [r["score"] for r in results if r.get("score") is not None]
    return {
        "total":        total,
        "correct":      correct,
        "accuracy":     round(correct / total, 4) if total else None,
        "mean_score":   round(statistics.mean(scores), 4) if scores else None,
        "sql_gen_rate": round(sum(1 for r in results if r["sql_generated"]) / total, 4) if total else None,
        "sql_exec_rate":round(sum(1 for r in results if r["sql_executed"])  / total, 4) if total else None,
        "timeouts":     sum(1 for r in results if "timeout" in r.get("generation_error", "")),
    }


def build_report(
    all_results: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    # group by query_type → stem
    by_qt: dict[str, dict[str, list]] = {}
    for r in all_results:
        qt   = r.get("query_type", "unknown")
        stem = r.get("source_stem", "unknown")
        by_qt.setdefault(qt, {}).setdefault(stem, []).append(r)

    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("  EXTENDED EVALUATION REPORT")
    lines.append("=" * 72)

    json_out: dict[str, Any] = {"by_query_type": {}}

    for qt in sorted(by_qt):
        stems     = by_qt[qt]
        qt_results = [r for stem_rs in stems.values() for r in stem_rs]
        qt_summary = build_stem_summary(qt_results)

        lines.append(f"\n{'─'*72}")
        lines.append(f"  {qt.upper()}")
        lines.append(f"{'─'*72}")
        lines.append(
            f"  {'Stem':<40} {'N':>5} {'Correct':>8} {'Acc':>7} {'MeanScore':>10}"
        )
        lines.append(f"  {'-'*40} {'-----':>5} {'-------':>8} {'------':>7} {'---------':>10}")

        qt_json: dict[str, Any] = {"stems": {}, "total": qt_summary}
        for stem in sorted(stems):
            s   = build_stem_summary(stems[stem])
            acc = f"{s['accuracy']:.3f}" if s["accuracy"] is not None else "  n/a"
            ms  = f"{s['mean_score']:.3f}" if s["mean_score"] is not None else "  n/a"
            lines.append(
                f"  {stem:<40} {s['total']:>5} {s['correct']:>8} {acc:>7} {ms:>10}"
            )
            qt_json["stems"][stem] = s

        lines.append(f"  {'─'*40} {'─────':>5} {'───────':>8} {'──────':>7} {'─────────':>10}")
        acc = f"{qt_summary['accuracy']:.3f}" if qt_summary["accuracy"] is not None else "  n/a"
        ms  = f"{qt_summary['mean_score']:.3f}" if qt_summary["mean_score"] is not None else "  n/a"
        lines.append(
            f"  {'TOTAL':<40} {qt_summary['total']:>5} {qt_summary['correct']:>8} {acc:>7} {ms:>10}"
        )
        lines.append(
            f"  SQL gen={qt_summary['sql_gen_rate']:.2%}  "
            f"exec={qt_summary['sql_exec_rate']:.2%}  "
            f"timeouts={qt_summary['timeouts']}"
        )
        json_out["by_query_type"][qt] = qt_json

    # overall
    overall = build_stem_summary(all_results)
    lines.append(f"\n{'='*72}")
    lines.append("  OVERALL")
    lines.append(f"{'='*72}")
    lines.append(f"  Total questions : {overall['total']}")
    lines.append(f"  Correct         : {overall['correct']}")
    lines.append(f"  Accuracy        : {overall['accuracy']:.3f}" if overall["accuracy"] is not None else "  Accuracy: n/a")
    lines.append(f"  SQL gen rate    : {overall['sql_gen_rate']:.2%}" if overall["sql_gen_rate"] else "")
    lines.append(f"  SQL exec rate   : {overall['sql_exec_rate']:.2%}" if overall["sql_exec_rate"] else "")
    lines.append(f"  Timeouts        : {overall['timeouts']}")

    json_out["overall"] = overall

    report_text = "\n".join(lines)
    print(report_text)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation_report.txt").write_text(report_text, encoding="utf-8")
    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(json_out, indent=2), encoding="utf-8"
    )
    # per-question detail
    (output_dir / "evaluation_details.json").write_text(
        json.dumps(all_results, indent=2), encoding="utf-8"
    )
    print(f"\nSaved to {output_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate raster/raster-vector/extended + vector-only results."
    )
    parser.add_argument(
        "--cache-dir",
        default="baselines/cache",
        help="Root cache dir from run_raster_text2sql.py.",
    )
    parser.add_argument(
        "--provider",
        default="gemini",
        help="Provider prefix used in cache dir names (default: gemini).",
    )
    parser.add_argument(
        "--query-types",
        nargs="+",
        default=["raster_only", "raster_vector", "extended"],
        help="Which query type subdirs to evaluate.",
    )
    parser.add_argument(
        "--vector-csv",
        default=None,
        help=(
            "Path to parsed_eval CSV from baselines.py for vector-only queries. "
            "If omitted, vector-only evaluation is skipped."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="baselines/evaluation",
        help="Where to write evaluation_report.txt and evaluation_summary.json.",
    )
    parser.add_argument(
        "--source-stems-file",
        default=None,
        help="Optional newline-delimited allowlist of source stems to evaluate.",
    )
    return parser.parse_args()


def main() -> None:
    args      = parse_args()
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)

    print(f"Loading raster results from {cache_dir} ...")
    all_results: list[dict[str, Any]] = []

    # --- raster / raster-vector / extended ---
    cache_records = load_cache_results(cache_dir, args.provider, args.query_types)
    if args.source_stems_file:
        allowed_stems = {
            line.strip()
            for line in Path(args.source_stems_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        cache_records = [
            record for record in cache_records
            if record.get("source_stem") in allowed_stems
        ]
    print(f"  {len(cache_records)} questions loaded")

    for rec in cache_records:
        eval_result = evaluate_question(rec)
        all_results.append(eval_result)

    # --- vector-only (optional) ---
    if args.vector_csv:
        vec_path = Path(args.vector_csv)
        print(f"Loading vector-only results from {vec_path} ...")
        vec_rows = load_vector_csv(vec_path)
        print(f"  {len(vec_rows)} rows loaded")
        for row in vec_rows:
            all_results.append(evaluate_vector_row(row))

    if not all_results:
        print("No results found. Check --cache-dir and --query-types.")
        return

    print(f"\nEvaluating {len(all_results)} questions ...\n")
    build_report(all_results, output_dir)


if __name__ == "__main__":
    main()
