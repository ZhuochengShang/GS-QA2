import json
import ast
import os
import re
import subprocess
import csv
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple, List
import urllib.parse
import math
import sys
import shutil
import time

FAIL_PHRASE = "Failed to execute and debug the code within 5 times."

BASELINES_DIR = os.environ.get("GSQA_BASELINES_DIR", "")
if BASELINES_DIR not in sys.path:
    sys.path.append(BASELINES_DIR)
try:
    import evaluate as gsqa_eval  # type: ignore
except Exception:
    gsqa_eval = None


def run_task(task):
    env = os.environ.copy()
    env.clear()
    env.update(task["env"])

    proc = subprocess.Popen(
        task["cmd"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    stdout = []
    for line in proc.stdout:
        stdout.append(line)
        print(line, end="")

    proc.wait()
    return proc.returncode, "".join(stdout)


def _extract_last_json_object(stdout_text: str) -> Optional[Dict[str, Any]]:
    """
    Try to extract the last JSON object printed in stdout.
    Works even if the agent prints logs before/after.
    """
    lines = [ln for ln in stdout_text.splitlines() if ln.strip()]
    if not lines:
        return None

    # Fast path: last non-empty line is JSON
    last = lines[-1].strip()
    try:
        obj = json.loads(last)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Fallback: look for "Output: { ... }" style Python dict and parse it
    for ln in reversed(lines):
        if ln.startswith("Output:"):
            payload = ln[len("Output:"):].strip()
            try:
                obj = json.loads(payload)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                try:
                    obj = ast.literal_eval(payload)
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    pass

    # Robust path: scan for the last JSON object by decoder walk
    decoder = json.JSONDecoder()
    text = "\n".join(lines)
    last_obj = None
    idx = 0
    while True:
        start = text.find("{", idx)
        if start == -1:
            break
        try:
            obj, end = decoder.raw_decode(text, start)
            if isinstance(obj, dict):
                last_obj = obj
            idx = end
        except json.JSONDecodeError:
            idx = start + 1
    return last_obj


def _load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    if not path:
        return []
    if path.endswith(".jsonl"):
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        items.append(obj)
                except Exception:
                    continue
        return items
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        tasks = data.get("tasks")
        if isinstance(tasks, list):
            return tasks
    return []


def _parse_point_wkt(wkt: str) -> Optional[Tuple[float, float]]:
    if not isinstance(wkt, str):
        return None
    m = re.match(r"^POINT\s*\(\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*\)$", wkt.strip(), re.IGNORECASE)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def _parse_point_any(geom: Any) -> Optional[Tuple[float, float]]:
    if isinstance(geom, str):
        return _parse_point_wkt(geom)
    if isinstance(geom, dict):
        if geom.get("type") == "Point":
            coords = geom.get("coordinates")
            if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                try:
                    return float(coords[0]), float(coords[1])
                except Exception:
                    return None
        return None
    if isinstance(geom, (list, tuple)) and len(geom) >= 2:
        try:
            return float(geom[0]), float(geom[1])
        except Exception:
            return None
    return None


def _haversine_m(lon1, lat1, lon2, lat2) -> float:
    r = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _best_name_f1(pred_answers: List[Dict[str, Any]], true_answers: List[Dict[str, Any]]) -> Optional[float]:
    if not pred_answers or not true_answers:
        return None
    best = None
    pred_name_keys = ("name", "poi_name", "park_name", "road_name", "region_name", "lake_name")
    true_name_keys = ("name", "poi_name", "park_name", "road_name", "region_name", "lake_name")
    for p in pred_answers:
        p_name = None
        for k in pred_name_keys:
            p_name = p.get(k)
            if p_name:
                break
        if not p_name:
            # Handle nested tagsMap or flattened tagsMap/name key
            tags_map = p.get("tagsMap") if isinstance(p.get("tagsMap"), dict) else None
            if tags_map:
                p_name = tags_map.get("name")
        if not p_name and "tagsMap/name" in p:
            p_name = p.get("tagsMap/name")
        if not p_name and gsqa_eval:
            p_name = gsqa_eval.get_osm_value(p, "name")
        if not p_name:
            continue
        for t in true_answers:
            t_name = None
            for k in true_name_keys:
                t_name = t.get(k)
                if t_name:
                    break
            if not t_name and gsqa_eval:
                t_name = gsqa_eval.get_osm_value(t, "name")
            if not t_name:
                continue
            if gsqa_eval:
                _, _, f1 = gsqa_eval.evaluate_entity_names(str(p_name), str(t_name))
            else:
                f1 = _simple_name_f1(str(p_name), str(t_name))
            if best is None or f1 > best:
                best = f1
    return best


def _simple_name_f1(a: str, b: str) -> float:
    def _norm(s: str) -> str:
        s = s.lower()
        s = re.sub(r"[^a-z0-9\s]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s
    na = _norm(a)
    nb = _norm(b)
    if not na or not nb:
        return 0.0
    ta = na.split()
    tb = nb.split()
    sa = set(ta)
    sb = set(tb)
    if not sa or not sb:
        return 0.0
    tp = len(sa & sb)
    if tp == 0:
        return 0.0
    precision = tp / len(sa)
    recall = tp / len(sb)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _best_geom_dist_m(pred_answers: List[Dict[str, Any]], true_answers: List[Dict[str, Any]]) -> Optional[float]:
    best = None
    for p in pred_answers:
        p_geom = p.get("geometry")
        p_pt = _parse_point_any(p_geom) if p_geom is not None else None
        if not p_pt:
            continue
        for t in true_answers:
            t_geom = t.get("geometry")
            t_pt = _parse_point_any(t_geom) if t_geom is not None else None
            if not t_pt:
                continue
            dist = _haversine_m(p_pt[0], p_pt[1], t_pt[0], t_pt[1])
            if best is None or dist < best:
                best = dist
    return best


def _extract_last_json_from_file(jsonl_path: str) -> Optional[Dict[str, Any]]:
    if not jsonl_path or not os.path.exists(jsonl_path):
        return None
    last = None
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    last = obj
            except Exception:
                continue
    return last


def _validate_answers_json(result_obj: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Generic GS-QA validation:
    - require an "answers" list (can be empty)
    - if present, items should be objects
    - if "geometry" appears, it should be WKT POINT in lon/lat order
    """
    errs = []
    name_keys = ("name",)

    if "answers" not in result_obj or not isinstance(result_obj["answers"], list):
        errs.append("missing_or_invalid_answers_list")
        return False, errs

    for i, ans in enumerate(result_obj["answers"]):
        if not isinstance(ans, dict):
            errs.append(f"answer_item_not_object:{i}")
            continue
        if "name" not in ans:
            errs.append(f"name_field_missing:{i}")
            continue
        if "addr_state" not in ans:
            errs.append(f"addr_state_missing:{i}")
            continue
        for k in name_keys:
            if k in ans and (ans[k] is None or (isinstance(ans[k], str) and not ans[k].strip())):
                errs.append(f"name_field_empty:{i}:{k}")
                break
        if "addr_state" in ans and (ans["addr_state"] is None or (isinstance(ans["addr_state"], str) and not ans["addr_state"].strip())):
            errs.append(f"addr_state_empty:{i}")
        for k in ("poi_name", "park_name", "road_name", "region_name", "lake_name"):
            if k in ans:
                errs.append(f"unexpected_name_field:{i}:{k}")
                break
        if "geometry" in ans and ans["geometry"] is not None:
            if not isinstance(ans["geometry"], str):
                errs.append(f"geometry_not_string:{i}")
                continue
            geom = ans["geometry"].strip()
            if not re.search(r"^POINT\s*\(\s*-?\d+(\.\d+)?\s+-?\d+(\.\d+)?\s*\)$", geom, re.IGNORECASE):
                errs.append(f"geometry_not_wkt_point:{i}")

    return (len(errs) == 0), errs


def evaluate(stdout_text, expected_patterns, expected_paths, json_output_path: Optional[str] = None, ground_truth: Optional[List[Dict[str, Any]]] = None):
    errors = []
    if FAIL_PHRASE in stdout_text:
        errors.append("agent_failed_after_5_retries")

    # Keep your regex checks if you want them (optional)
    missing_patterns = []
    for pat in expected_patterns:
        if not re.search(pat, stdout_text, re.IGNORECASE):
            missing_patterns.append(pat)

    missing_paths = []
    for p in expected_paths:
        if not os.path.exists(p):
            missing_paths.append(p)

    # JSON extraction + schema validation
    parsed = _extract_last_json_from_file(json_output_path) if json_output_path else None
    if parsed is None:
        parsed = _extract_last_json_object(stdout_text)
    if parsed is None:
        errors.append("no_json_found_in_stdout")
    else:
        ok, json_errs = _validate_answers_json(parsed)
        if not ok:
            errors.extend([f"json_invalid:{e}" for e in json_errs])
        elif ground_truth:
            pred_answers = parsed.get("answers", []) if isinstance(parsed, dict) else []
            if not pred_answers:
                errors.append("empty_answers_for_nonempty_ground_truth")

    success = not errors and not missing_patterns and not missing_paths
    name_f1 = None
    geom_dist_m = None
    if parsed is not None and ground_truth is not None:
        pred_answers = parsed.get("answers", []) if isinstance(parsed, dict) else []
        if isinstance(pred_answers, list):
            name_f1 = _best_name_f1(pred_answers, ground_truth)
            geom_dist_m = _best_geom_dist_m(pred_answers, ground_truth)
    usage = None
    cost_estimate = None
    if parsed and isinstance(parsed, dict):
        usage = parsed.get("_usage")
        cost_estimate = parsed.get("_cost_estimate")
    return success, errors, missing_patterns, missing_paths, name_f1, geom_dist_m, usage, cost_estimate


def _safe_utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _extract_generated_code(stdout_text: str) -> Optional[str]:
    """
    Look for the latest URL-encoded code emitted by SpatialAnalysisAgent_headless.py.
    """
    prefixes = ("CODE_READY_URLENCODED2:", "CODE_READY_URLENCODED:")
    lines = [ln.strip() for ln in stdout_text.splitlines() if ln.strip()]
    for line in reversed(lines):
        for prefix in prefixes:
            if line.startswith(prefix):
                return urllib.parse.unquote(line[len(prefix):])
    return None


def _write_log_files(logs_dir: str, task: Dict[str, Any], stdout_text: str, result: Dict[str, Any]) -> None:
    os.makedirs(logs_dir, exist_ok=True)
    ts = _safe_utc_ts()
    task_name = task.get("name", "task")
    base = f"{task_name}_{ts}"

    log_path = os.path.join(logs_dir, f"{base}.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"task={task_name}\n")
        f.write(f"timestamp_utc={result.get('timestamp_utc')}\n")
        f.write("cmd=" + " ".join(task.get("cmd", [])) + "\n\n")
        if task.get("json_output_path"):
            f.write(f"json_output_path={task['json_output_path']}\n")
        f.write(stdout_text)

    result_path = os.path.join(logs_dir, f"{base}.result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    json_output_path = task.get("json_output_path")
    if json_output_path and os.path.exists(json_output_path):
        jsonl_copy_path = os.path.join(logs_dir, f"{base}.output.jsonl")
        shutil.copyfile(json_output_path, jsonl_copy_path)

    code = _extract_generated_code(stdout_text)
    if code:
        code_path = os.path.join(logs_dir, f"{base}.generated.py")
        with open(code_path, "w", encoding="utf-8") as f:
            f.write(code)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Auto-eval runner for GS-QA tasks.")
    parser.add_argument("--tasks-json", help="Path to tasks JSON file", default=None)
    parser.add_argument("--task-id", help="Select a task by id", default=None)
    parser.add_argument("--category", help="Select a task by category (e.g., T1)", default=None)
    parser.add_argument("--all", action="store_true", help="Run all tasks in the JSON file")
    parser.add_argument("--model", help="Model name for SpatialAnalysisAgent_headless.py", default="gpt-4o")
    parser.add_argument("--run-only", action="store_true", help="Run tasks and save outputs, skip evaluation scoring")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate existing outputs only, do not run model calls")
    args = parser.parse_args()
    if args.run_only and args.eval_only:
        raise ValueError("Use only one of --run-only or --eval-only.")

    DATA_PROMPT = r"""
[DATA_PROMPT]
You are given OSM data in GeoJSON format under folder OsmData.

Directory structure (relative to the input data path):
- parks/part-*.geojson
- pois/part-*.geojson
- lakes/part-*.geojson
- postal_codes/   (contains many files like part-00000*.geojson)
- roads/part-*.geojson

DATA ACCESS RULES (CRITICAL):
- Each layer is split across many indexed shard files named `part-*.geojson`.
- The GeoJSON shards are very large. Do NOT load full layers with `geopandas.read_file` for every shard.
- Do NOT concatenate all shards into one GeoDataFrame unless the task explicitly requires every geometry.
- Stream GeoJSON features with `ijson`, filter each feature immediately, and only convert the small filtered result set to Shapely/GeoPandas when geometry operations are needed.
- Stop early when finding a named reference entity.
- Do NOT generate, mock, or write any dummy GeoJSON/CSV files.
- Do NOT change filenames, folder names, or invent alternative paths.
- The canonical file patterns are exactly:
  - `glob.glob(os.path.join(data_root, "pois", "part-*.geojson"))`
  - `glob.glob(os.path.join(data_root, "parks", "part-*.geojson"))`
  - `glob.glob(os.path.join(data_root, "lakes", "part-*.geojson"))`
  - `glob.glob(os.path.join(data_root, "roads", "part-*.geojson"))`
  - `glob.glob(os.path.join(data_root, "postal_codes", "part-*.geojson"))`
- Do NOT use shortcut filenames like `pois.geojson`, `parks.geojson`, `lakes.geojson`, `roads.geojson`, `postal_codes.geojson`, `pois.shp`, or `pois.*`.
- Do NOT open a literal wildcard path. Always expand with `glob.glob` first.
- Use this streaming pattern for large layers:
```python
import glob
import ijson
import os
from shapely.geometry import shape

def iter_features(data_root, layer):
    files = sorted(glob.glob(os.path.join(data_root, layer, "part-*.geojson")))
    for path in files:
        with open(path, "rb") as f:
            for feat in ijson.items(f, "features.item"):
                yield feat

def get_prop(feat, key):
    props = feat.get("properties") or {}
    v = props.get(key)
    if v is not None and str(v).strip() != "":
        return v
    tags = props.get("tagsMap") or {}
    if isinstance(tags, dict):
        v = tags.get(key)
        if v is not None and str(v).strip() != "":
            return v
    return None

def norm_name(value):
    return " ".join(str(value).lower().split())

def find_by_name(data_root, layer, target_name):
    target = norm_name(target_name)
    fallback = None
    for feat in iter_features(data_root, layer):
        name = get_prop(feat, "name") or get_prop(feat, "uname")
        if not name:
            continue
        n = norm_name(name)
        if n == target:
            return feat
        if fallback is None and target in n:
            fallback = feat
    return fallback

def feature_geometry(feat):
    geom = feat.get("geometry")
    return shape(geom) if geom else None
```
- If a required file is missing, return {"answers": []}.

IMPORTANT DATA MODEL:
- In the POIs file, many attributes are inside a nested dict column called "tagsMap".
- Read `tagsMap/<key>` as `tagsMap["<key>"]`.
- Always resolve fields in this order:
  1) top-level column (e.g., `name`, `amenity`)
  2) `tagsMap[<same_key>]`
  3) null if both are missing
- Do not invent columns or keys.

Recommended helper pattern:
```python
def pick_attr(row, key):
    v = row.get(key, None)
    if v is not None and str(v).strip() != "":
        return v
    t = row.get("tagsMap", None)
    if isinstance(t, dict):
        tv = t.get(key, None)
        if tv is not None and str(tv).strip() != "":
            return tv
    return None
```

Tables / Schemas:

Table 1: POIs (pois/part-*.geojson shards)
Contains points of interest.
Common top-level columns:
- id: unique identifier
- geometry: GeoJSON geometry
- name: name of the POI (if missing, fallback to tagsMap/name)
- amenity: type of amenity (restaurants are amenity="restaurant"; fallback to tagsMap/amenity)

Common tagsMap fields (examples; may be missing per feature):
- tagsMap/name
- tagsMap/wikidata
- tagsMap/wikipedia
- tagsMap/addr_state
- tagsMap/addr_city
- tagsMap/cuisine
- tagsMap/leisure
- tagsMap/tourism
- tagsMap/takeaway
- tagsMap/drive_through
- tagsMap/museum
- tagsMap/healthcare
- tagsMap/outdoor_seating
- tagsMap/emergency
- tagsMap/restaurant

Note:
- Galleries are tagged as tagsMap/tourism = "gallery" (not amenity).

Table 2: Parks (parks/part-*.geojson shards)
Contains parks, gardens, etc.
Columns may be top-level or inside tagsMap. Use top-level first, then tagsMap:
- id, geometry, uname, wikidata, wikipedia, leisure, park, tourism

Table 3: Roads (roads/part-*.geojson shards)
Contains roads, walkways, etc.
Contains road geometries (LineString/MultiLineString).
Columns may be top-level or inside tagsMap. Use top-level first, then tagsMap:
- id, geometry, uname, wikidata, wikipedia, highway, sidewalk, foot, bicycle, cycleway

Table 4: Regions (postal_codes/part-*.geojson shards)
Contains administrative region boundaries, like cities and states, etc.
Columns may include:
- id, geometry, uname, border_type, wikidata, wikipedia
""".strip()

    DEFAULT_TASK = r"""
[TASK]
Can you find me a restaurant that's 150 kilometers from New Almaden Mining Museum, CA?
""".strip()

    CONSTRAINTS = r"""
[CONSTRAINTS]
1) Use ONLY the provided GeoJSON files (no external APIs or geocoding).
2) Do NOT import any QGIS modules (e.g., do not import from qgis.core, processing).
3) Use standard Python geospatial libraries only (e.g., geopandas/shapely/pyproj).
3.1) Never call `processing.run` or any `native:`/`qgis:` algorithm.
3.2) Do NOT create, write, or simulate dummy input datasets/files. Read only from the provided data path.
3.3) If your draft code includes any qgis/processing import, rewrite it before execution.
3.4) Use exact OSM file patterns from [DATA_PROMPT]; never substitute with `pois.geojson`, `parks.geojson`, `roads.geojson`, `pois.shp`, `pois.*`, or other single-file shortcuts.
3.5) PERFORMANCE: Do NOT read all POI, parks, lakes, roads, or postal_codes shards into GeoPandas. Do NOT concatenate full layers. Use `ijson` streaming to scan features and filter by name/type/state while streaming. Only build a GeoDataFrame from the small filtered candidate set.
4) Ensure distance calculation is in meters (use an appropriate projection).
5) Do NOT hard-code coordinates or geocode by guesswork. The reference entity must be located by searching the provided GeoJSON data.
6) You must find the reference entity by matching its name from the resolved "name" field (top-level name first, then tagsMap/name), and use its geometry as the reference point (no guessed coordinates). Do not use external fuzzy-matching libraries.
   Example pattern:
   ref_feat = find_by_name(data_root, "pois", "<POI name only>") or find_by_name(data_root, "parks", "<POI name only>") or find_by_name(data_root, "lakes", "<POI name only>")
   ref_geom = feature_geometry(ref_feat)
   If exact match fails, use case-insensitive substring contains. Do not use fuzzy libraries.
7) Do NOT use eval on tagsMap; treat it as a dict. Convert GeoJSON geometry dictionaries with `shapely.geometry.shape()` only after the feature has been filtered/selected.
8) Only call .to_crs() on GeoDataFrame/GeoSeries, never on a single geometry.
""".strip()

    OUTPUT_SPEC = r"""
[OUTPUT_SPEC]
- Print ONLY valid JSON using print(json.dumps(result)) (no explanations, no markdown, no prefixes like "Output:", no extra prints).
- Geometry must be WKT in EPSG:4326 (lon/lat order). If you compute in a projected CRS, reproject to EPSG:4326 before output.
- Example reprojection snippet:
```
gdf = gdf.to_crs("EPSG:4326")
gdf["geometry_wkt"] = gdf.geometry.apply(lambda g: g.wkt if g is not None else None)
```
- When populating the output "geometry" field, use the geometry's `.wkt` after reprojection (do not call `.to_crs()` or `.to_wkt()` on a single geometry).
- Use geometry_wkt when populating the output "geometry" field.

Output schema must MATCH the ground‑truth format:
{
  "answers": [
    {
      "id": <int>,
      "osm_id": <int or null>,
      "geometry": "<WKT POINT>",
      // plus ONLY the fields relevant to the entity type / question:
      //   name (use tagsMap/name when present)
      //   addr_state (use tagsMap/addr_state when present)
      //   optional category fields: amenity | tourism | leisure | highway | waterway | water
      //   optional measurement field when asked: angle (degrees) OR distance (meters)
    }
  ]
}

Rules:
- Use output key "name" only (NOT poi_name/park_name/etc).
- Resolve value of "name" as: top-level `name` first, else `tagsMap["name"]`.
- The "name" field must be non-null and non-empty.
- If your selected feature has no name, skip it and select the next best candidate that does.
- Resolve `addr_state` as top-level `addr_state` first, else `tagsMap["addr_state"]`.
- Include addr_state in outputs when available; if addr_state is missing, skip that candidate.
- For string matching, try exact first, then case-insensitive exact, then substring contains. Do not use fuzzy libraries.
- When computing distances, reproject BOTH the reference geometry and candidate geometries to the same projected CRS before measuring.
- For location/address questions, output textual location in key `address`.
- For direction/angle questions, output numeric angle in degrees in key `angle`.
- Use angle in degrees when direction/angle is requested.
- Use distance in meters when distance is requested.
- Do NOT include any extra keys beyond what appears in ground‑truth style (no additional fields).
- If no results satisfy constraints, return {"answers": []}.
- If code execution fails, still print valid JSON with {"answers": []} and avoid any non-JSON output.

""".strip()

    def _load_task_from_json(path: str, task_id: Optional[str], category: Optional[str]) -> Optional[Dict[str, Any]]:
        if not path:
            return None
        data = _load_json_or_jsonl(path)
        if not isinstance(data, list):
            return None
        if task_id:
            for item in data:
                if isinstance(item, dict) and item.get("id") == task_id:
                    return item
            return None
        if category:
            for item in data:
                if isinstance(item, dict) and item.get("category") == category:
                    return item
            return None
        return None

    def _load_tasks_list(path: str) -> List[Dict[str, Any]]:
        if not path:
            return []
        data = _load_json_or_jsonl(path)
        return data if isinstance(data, list) else []

    def _build_task_prompt(task_text: str) -> str:
        task_block = f"[TASK]\n{task_text}".strip()
        return "\n\n".join([DATA_PROMPT, task_block, CONSTRAINTS, OUTPUT_SPEC])

    tasks = []
    if args.all and args.tasks_json:
        source_items = _load_tasks_list(args.tasks_json)
        source_stem = os.path.splitext(os.path.basename(args.tasks_json))[0]
        for idx, item in enumerate(source_items):
            if not isinstance(item, dict):
                continue
            task_text = item.get("question") or item.get("task") or item.get("task_text") or ""
            if not task_text:
                continue
            category = item.get("category") or "T"
            task_id = item.get("id") or item.get("task_ID") or str(idx)
            name = f"{category}_{task_id}" if item.get("id") else f"{source_stem}_{task_id}"
            tasks.append({"name": name, "task_text": task_text, "ground_truth": item.get("answers")})
    else:
        loaded_task = _load_task_from_json(args.tasks_json, args.task_id, args.category)
        if loaded_task:
            task_text = loaded_task.get("question") or loaded_task.get("task") or loaded_task.get("task_text") or ""
            category = loaded_task.get("category") or "T"
            task_id = loaded_task.get("id") or loaded_task.get("task_ID") or "task"
            name = f"{category}_{task_id}"
        else:
            task_text = DEFAULT_TASK.split("\n", 1)[1]
            name = "restaurant"
        tasks.append({"name": name, "task_text": task_text, "ground_truth": loaded_task.get("answers") if loaded_task else None})
    base_env = {
        "HOME": os.environ.get("HOME", ""),
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "QGIS_PREFIX_PATH": "/Applications/QGIS.app",
        "PROJ_LIB": "/Applications/QGIS.app/Contents/Resources/qgis/proj",
        "GDAL_DATA": "/Applications/QGIS.app/Contents/Resources/gdal",
        "DYLD_LIBRARY_PATH": "/Applications/QGIS.app/Contents/Frameworks:/Applications/QGIS.app/Contents/MacOS",
        "QT_QPA_PLATFORM": "offscreen",
        "PYTHONPATH": "/Applications/QGIS.app/Contents/Resources/python:/Applications/QGIS.app/Contents/Resources/python/plugins:/Applications/QGIS.app/Contents/Resources/qgis/python",
    }
    # Pass through common API keys for model providers.
    if os.environ.get("OPENAI_API_KEY"):
        base_env["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]
    if os.environ.get("GOOGLE_API_KEY"):
        base_env["GOOGLE_API_KEY"] = os.environ["GOOGLE_API_KEY"]
    if os.environ.get("GEMINI_API_KEY"):
        base_env["GEMINI_API_KEY"] = os.environ["GEMINI_API_KEY"]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.environ.get("SPATIALAGENT_WORKSPACE") or os.path.join(script_dir, "workspace")
    data_path = os.environ.get("SPATIALAGENT_DATA_PATH") or "/local_data/scratch/zshan011/osm/osm_extract"
    artifacts_dir = os.path.join(workspace_dir, "task_artifacts")
    outputs_dir = os.path.join(artifacts_dir, "outputs")
    python_bin = os.environ.get("SPATIALAGENT_PYTHON") or sys.executable

    for t in tasks:
        task_prompt = _build_task_prompt(t["task_text"])
        t["env"] = base_env
        t["cmd"] = [
            python_bin,
            os.path.join(script_dir, "SpatialAnalysisAgent_headless.py"),
            "--task",
            task_prompt,
            "--task-name",
            t["name"],
            "--data-path",
            data_path,
            "--model",
            args.model,
            "--workspace",
            workspace_dir,
            "--review",
            "true",
            "--reasoning-effort",
            "medium",
        ]
        t["json_output_path"] = os.path.join(outputs_dir, f"{t['name']}.output.jsonl")
        t["expected_patterns"] = []
        t["expected_paths"] = []

    results = []
    for task in tasks:
        print(f"\n=== Running task: {task['name']} ===")
        task_started_at = time.perf_counter()
        run_seconds = 0.0
        eval_seconds = 0.0
        if args.eval_only:
            returncode = 0
            stdout_text = ""
        else:
            run_started_at = time.perf_counter()
            returncode, stdout_text = run_task(task)
            run_seconds = time.perf_counter() - run_started_at

        if args.run_only:
            result = {
                "name": task["name"],
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "returncode": returncode,
                "success": returncode == 0,
                "errors": [],
                "missing_patterns": [],
                "missing_paths": [],
                "name_f1": None,
                "geom_dist_m": None,
                "ground_truth_loaded": task.get("ground_truth") is not None,
                "usage": None,
                "cost_estimate": None,
                "mode": "run_only",
                "run_seconds": run_seconds,
                "eval_seconds": eval_seconds,
                "total_seconds": time.perf_counter() - task_started_at,
            }
        else:
            eval_started_at = time.perf_counter()
            success, errors, missing_patterns, missing_paths, name_f1, geom_dist_m, usage, cost_estimate = evaluate(
                stdout_text,
                task.get("expected_patterns", []),
                task.get("expected_paths", []),
                json_output_path=task.get("json_output_path"),
                ground_truth=task.get("ground_truth"),
            )
            eval_seconds = time.perf_counter() - eval_started_at

            result = {
                "name": task["name"],
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "returncode": returncode,
                "success": success and returncode == 0,
                "errors": errors,
                "missing_patterns": missing_patterns,
                "missing_paths": missing_paths,
                "name_f1": name_f1,
                "geom_dist_m": geom_dist_m,
                "ground_truth_loaded": task.get("ground_truth") is not None,
                "usage": usage,
                "cost_estimate": cost_estimate,
                "mode": "eval_only" if args.eval_only else "run_and_eval",
                "run_seconds": run_seconds,
                "eval_seconds": eval_seconds,
                "total_seconds": time.perf_counter() - task_started_at,
            }
        results.append(result)

        logs_dir = os.path.join(os.path.dirname(__file__), "logs")
        _write_log_files(logs_dir, task, stdout_text, result)

        print("\n=== TASK RESULT ===")
        print(json.dumps(result, indent=2))

    print("\n=== ALL RESULTS ===")
    print(json.dumps(results, indent=2))

    logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    json_summary_path = os.path.join(logs_dir, f"summary_{_safe_utc_ts()}.json")
    with open(json_summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON summary written to: {json_summary_path}")

    csv_path = os.path.join(logs_dir, f"summary_{_safe_utc_ts()}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "timestamp_utc",
                "returncode",
                "success",
                "errors",
                "missing_patterns",
                "missing_paths",
                "name_f1",
                "geom_dist_m",
                "run_seconds",
                "eval_seconds",
                "total_seconds",
                "total_tokens",
                "estimated_cost_usd",
            ],
        )
        writer.writeheader()
        for r in results:
            total_tokens = None
            estimated_cost = None
            usage = r.get("usage") or {}
            cost = r.get("cost_estimate") or {}
            if isinstance(usage, dict):
                total_tokens = usage.get("total_tokens")
            if isinstance(cost, dict):
                estimated_cost = sum(
                    (v.get("estimated_cost_usd", 0) or 0) for v in cost.values() if isinstance(v, dict)
                )
            writer.writerow(
                {
                    "name": r.get("name"),
                    "timestamp_utc": r.get("timestamp_utc"),
                    "returncode": r.get("returncode"),
                    "success": r.get("success"),
                    "errors": ";".join(r.get("errors", [])),
                    "missing_patterns": ";".join(r.get("missing_patterns", [])),
                    "missing_paths": ";".join(r.get("missing_paths", [])),
                    "name_f1": r.get("name_f1"),
                    "geom_dist_m": r.get("geom_dist_m"),
                    "run_seconds": r.get("run_seconds"),
                    "eval_seconds": r.get("eval_seconds"),
                    "total_seconds": r.get("total_seconds"),
                    "total_tokens": total_tokens,
                    "estimated_cost_usd": estimated_cost,
                }
            )
    print(f"\nCSV summary written to: {csv_path}")


if __name__ == "__main__":
    main()
