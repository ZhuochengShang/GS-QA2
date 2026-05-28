import argparse
import ast
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


FAIL_PHRASE = "Failed to execute and debug the code within 5 times."
SCRIPT_DIR = Path(__file__).resolve().parent
HEADLESS_PATH = SCRIPT_DIR / "SpatialAnalysisAgent_headless.py"
DEFAULT_WORKSPACE = SCRIPT_DIR / "workspace"
DEFAULT_LOGS_DIR = SCRIPT_DIR / "logs"


def _safe_utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _default_python_bin() -> str:
    venv_python = Path.home() / ".venvs" / "spatialagent" / "bin" / "python"
    return str(venv_python) if venv_python.exists() else sys.executable


def _load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        items: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    items.append(obj)
        return items

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict) and isinstance(data.get("tasks"), list):
        return [x for x in data["tasks"] if isinstance(x, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def _select_tasks(items: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.all:
        return items
    if args.task_id is not None:
        return [x for x in items if str(x.get("id")) == str(args.task_id)]
    if args.category is not None:
        return [x for x in items if str(x.get("category") or x.get("type")) == args.category]
    return items[:1]


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.:+-]+", "_", value.strip())
    return value.strip("_")[:120] or "task"


def _task_name(item: Dict[str, Any], idx: int, source_stem: str) -> str:
    category = item.get("category") or item.get("type") or source_stem
    task_id = item.get("id")
    if task_id is None:
        task_id = idx
    return _slug(f"{category}_{task_id}")


def _compact_json(obj: Any, max_chars: int = 20000) -> str:
    text = json.dumps(obj, ensure_ascii=True)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...<truncated>"


def _build_task_prompt(item: Dict[str, Any]) -> str:
    question = item.get("question") or item.get("task") or item.get("task_text") or ""
    answer_type = item.get("answer_type") or item.get("category") or item.get("type") or ""
    entities = item.get("question_entities") or {}

    return f"""
[DATA_PROMPT]
You are solving a raster or raster-vector GIS benchmark task.

The input data path supplied to the program may contain:
- DEM/elevation rasters, usually GeoTIFF files.
- slope rasters, if provided.
- vector files such as GeoJSON, GeoPackage, or shapefiles, if provided.

The benchmark record may also include authoritative geometry in `question_entities`.
When `question_entities` contains `geo_wkt`, use that WKT directly instead of geocoding or guessing.
Do not call external APIs. Do not invent coordinates, geometries, rasters, or vector files.

[TASK]
{question}

[BENCHMARK_CONTEXT]
answer_type: {answer_type}
question_entities: {_compact_json(entities)}

[CONSTRAINTS]
1) Use the provided raster/vector data path and the WKT geometries from BENCHMARK_CONTEXT.
2) Use Python geospatial libraries such as rasterio, numpy, shapely, pyproj, and geopandas.
3) Do not import qgis modules and do not call processing.run.
4) For point elevation/slope questions, sample the raster at the provided point WKT.
5) For polygon/region summaries, mask or window the raster by the provided polygon WKT and compute the requested statistic.
6) For route/line slope questions, sample elevations along the line and compute slope in degrees when requested.
7) For raster-vector queries, use vector records from the provided files when available; otherwise use entity details from BENCHMARK_CONTEXT only when they are sufficient to answer.
8) Reproject geometries to the raster CRS before sampling/masking.
9) Treat NoData values as missing.
10) If the available raster does not cover the requested geometry or required data is missing, return {{"answers": []}}.

[OUTPUT_SPEC]
Print only valid JSON with this top-level shape:
{{"answers": [...]}}

Use keys that match the benchmark answer style when possible:
- elevation
- slope_degrees or slope_deg
- count
- name or poi_name
- geometry
- addr_state

For numeric answers, output numbers, not formatted strings.
For feature answers, include a name field and geometry WKT when available.
If no result can be computed, print exactly {{"answers": []}}.
""".strip()


def _base_env() -> Dict[str, str]:
    env = {
        "HOME": os.environ.get("HOME", ""),
        "PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/sbin:/sbin"),
        "QT_QPA_PLATFORM": os.environ.get("QT_QPA_PLATFORM", "offscreen"),
    }
    for key in (
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "QGIS_PREFIX_PATH",
        "PROJ_LIB",
        "GDAL_DATA",
        "DYLD_LIBRARY_PATH",
        "LD_LIBRARY_PATH",
        "PYTHONPATH",
        "CONDA_PREFIX",
    ):
        if os.environ.get(key):
            env[key] = os.environ[key]

    if sys.platform == "darwin" and "QGIS_PREFIX_PATH" not in env:
        env.update(
            {
                "QGIS_PREFIX_PATH": "/Applications/QGIS.app",
                "PROJ_LIB": "/Applications/QGIS.app/Contents/Resources/qgis/proj",
                "GDAL_DATA": "/Applications/QGIS.app/Contents/Resources/gdal",
                "DYLD_LIBRARY_PATH": "/Applications/QGIS.app/Contents/Frameworks:/Applications/QGIS.app/Contents/MacOS",
                "PYTHONPATH": "/Applications/QGIS.app/Contents/Resources/python:/Applications/QGIS.app/Contents/Resources/python/plugins:/Applications/QGIS.app/Contents/Resources/qgis/python",
            }
        )
    return env


def _run_task(task: Dict[str, Any], env: Dict[str, str]) -> Tuple[int, str]:
    proc = subprocess.Popen(
        task["cmd"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    lines: List[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        lines.append(line)
        print(line, end="")
    proc.wait()
    return proc.returncode, "".join(lines)


def _extract_last_json_object(stdout_text: str) -> Optional[Dict[str, Any]]:
    lines = [ln.strip() for ln in stdout_text.splitlines() if ln.strip()]
    for line in reversed(lines):
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        if line.startswith("Output:"):
            payload = line[len("Output:") :].strip()
            try:
                obj = ast.literal_eval(payload)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
    return None


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


def _extract_generated_code(stdout_text: str) -> Optional[str]:
    for line in reversed([ln.strip() for ln in stdout_text.splitlines() if ln.strip()]):
        for prefix in ("CODE_READY_URLENCODED2:", "CODE_READY_URLENCODED:"):
            if line.startswith(prefix):
                return urllib.parse.unquote(line[len(prefix) :])
    return None


def _numbers_from_answers(answers: List[Dict[str, Any]]) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for ans in answers:
        if not isinstance(ans, dict):
            continue
        for key, value in ans.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                values.setdefault(key, float(value))
    return values


def _name_values(answers: List[Dict[str, Any]]) -> List[str]:
    keys = ("name", "poi_name", "road_name", "region_name", "park_name", "lake_name")
    out: List[str] = []
    for ans in answers:
        if not isinstance(ans, dict):
            continue
        for key in keys:
            value = ans.get(key)
            if value:
                out.append(str(value))
                break
    return out


def _simple_name_f1(a: str, b: str) -> float:
    def norm(s: str) -> List[str]:
        s = re.sub(r"[^a-z0-9\s]+", " ", s.lower())
        return [x for x in s.split() if x]

    ta = set(norm(a))
    tb = set(norm(b))
    if not ta or not tb:
        return 0.0
    overlap = len(ta & tb)
    if overlap == 0:
        return 0.0
    precision = overlap / len(ta)
    recall = overlap / len(tb)
    return 2 * precision * recall / (precision + recall)


def _best_name_f1(pred: List[Dict[str, Any]], truth: List[Dict[str, Any]]) -> Optional[float]:
    pred_names = _name_values(pred)
    truth_names = _name_values(truth)
    if not pred_names or not truth_names:
        return None
    return max(_simple_name_f1(p, t) for p in pred_names for t in truth_names)


def _numeric_score(
    pred: List[Dict[str, Any]],
    truth: List[Dict[str, Any]],
    abs_tol: float,
    rel_tol: float,
) -> Tuple[Optional[bool], Dict[str, Any]]:
    pred_numbers = _numbers_from_answers(pred)
    truth_numbers = _numbers_from_answers(truth)
    shared = sorted(set(pred_numbers) & set(truth_numbers))
    details: Dict[str, Any] = {"shared_numeric_keys": shared, "numeric_errors": {}}
    if not shared:
        return None, details

    ok = True
    for key in shared:
        p = pred_numbers[key]
        t = truth_numbers[key]
        err = abs(p - t)
        allowed = max(abs_tol, abs(t) * rel_tol)
        details["numeric_errors"][key] = {
            "pred": p,
            "truth": t,
            "abs_error": err,
            "allowed": allowed,
        }
        if err > allowed:
            ok = False
    return ok, details


def _evaluate(
    stdout_text: str,
    json_output_path: str,
    ground_truth: Optional[List[Dict[str, Any]]],
    abs_tol: float,
    rel_tol: float,
) -> Dict[str, Any]:
    errors: List[str] = []
    if FAIL_PHRASE in stdout_text:
        errors.append("agent_failed_after_5_retries")

    parsed = _extract_last_json_from_file(json_output_path) or _extract_last_json_object(stdout_text)
    if parsed is None:
        errors.append("no_json_found")
        return {
            "success": False,
            "errors": errors,
            "parsed": None,
            "numeric_match": None,
            "numeric_details": {},
            "name_f1": None,
        }

    answers = parsed.get("answers")
    if not isinstance(answers, list):
        errors.append("missing_or_invalid_answers_list")
        answers = []

    numeric_match = None
    numeric_details: Dict[str, Any] = {}
    name_f1 = None
    if ground_truth is not None:
        numeric_match, numeric_details = _numeric_score(answers, ground_truth, abs_tol, rel_tol)
        name_f1 = _best_name_f1(answers, ground_truth)

    if ground_truth is not None:
        scored = numeric_match is not None or name_f1 is not None
        if not scored:
            errors.append("no_comparable_ground_truth_fields")
        if numeric_match is False:
            errors.append("numeric_mismatch")

    return {
        "success": not errors,
        "errors": errors,
        "parsed": parsed,
        "numeric_match": numeric_match,
        "numeric_details": numeric_details,
        "name_f1": name_f1,
    }


def _write_log_files(logs_dir: Path, task: Dict[str, Any], stdout_text: str, result: Dict[str, Any]) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    base = f"{task['name']}_{_safe_utc_ts()}"

    log_path = logs_dir / f"{base}.log"
    log_path.write_text(stdout_text, encoding="utf-8")

    result_path = logs_dir / f"{base}.result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    json_output_path = task.get("json_output_path")
    if json_output_path and os.path.exists(json_output_path):
        shutil.copyfile(json_output_path, logs_dir / f"{base}.output.jsonl")

    code = _extract_generated_code(stdout_text)
    if code:
        (logs_dir / f"{base}.generated.py").write_text(code, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-eval runner for raster-only and raster-vector tasks.")
    parser.add_argument("--tasks-json", required=True, help="JSON or JSONL task file.")
    parser.add_argument("--data-path", required=True, help="Semicolon-separated raster/vector files or folders.")
    parser.add_argument("--all", action="store_true", help="Run all tasks in the task file.")
    parser.add_argument("--task-id", default=None, help="Run one task by id.")
    parser.add_argument("--category", default=None, help="Run tasks matching category/type.")
    parser.add_argument("--model", default="gpt-4o", help="Model name for SpatialAnalysisAgent_headless.py.")
    parser.add_argument("--python-bin", default=_default_python_bin(), help="Python interpreter for the headless runner.")
    parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE), help="Workspace for generated artifacts.")
    parser.add_argument("--logs-dir", default=str(DEFAULT_LOGS_DIR), help="Directory for eval logs/summaries.")
    parser.add_argument("--review", default="true", help="Pass-through review flag for the headless runner.")
    parser.add_argument("--reasoning-effort", default="medium", help="Pass-through reasoning effort.")
    parser.add_argument("--numeric-abs-tol", type=float, default=1.0, help="Absolute tolerance for numeric scoring.")
    parser.add_argument("--numeric-rel-tol", type=float, default=0.02, help="Relative tolerance for numeric scoring.")
    parser.add_argument("--run-only", action="store_true", help="Run tasks but skip scoring.")
    parser.add_argument("--eval-only", action="store_true", help="Score existing outputs without model calls.")
    args = parser.parse_args()

    if args.run_only and args.eval_only:
        raise SystemExit("Use only one of --run-only or --eval-only.")

    source_items = _load_json_or_jsonl(args.tasks_json)
    selected = _select_tasks(source_items, args)
    if not selected:
        raise SystemExit("No tasks selected.")

    workspace = Path(args.workspace)
    outputs_dir = workspace / "task_artifacts" / "outputs"
    logs_dir = Path(args.logs_dir)
    source_stem = Path(args.tasks_json).stem
    env = _base_env()

    results: List[Dict[str, Any]] = []
    for idx, item in enumerate(selected):
        name = _task_name(item, idx, source_stem)
        prompt = _build_task_prompt(item)
        json_output_path = str(outputs_dir / f"{name}.output.jsonl")
        task = {
            "name": name,
            "cmd": [
                args.python_bin,
                str(HEADLESS_PATH),
                "--task",
                prompt,
                "--task-name",
                name,
                "--data-path",
                args.data_path,
                "--model",
                args.model,
                "--workspace",
                str(workspace),
                "--review",
                args.review,
                "--reasoning-effort",
                args.reasoning_effort,
            ],
            "json_output_path": json_output_path,
            "ground_truth": item.get("answers") if isinstance(item.get("answers"), list) else None,
        }

        print(f"\n=== Running raster task: {name} ===")
        started = time.perf_counter()
        stdout_text = ""
        returncode = 0
        run_seconds = 0.0

        if not args.eval_only:
            run_started = time.perf_counter()
            returncode, stdout_text = _run_task(task, env)
            run_seconds = time.perf_counter() - run_started

        eval_result: Dict[str, Any]
        eval_seconds = 0.0
        if args.run_only:
            eval_result = {
                "success": returncode == 0,
                "errors": [],
                "parsed": None,
                "numeric_match": None,
                "numeric_details": {},
                "name_f1": None,
            }
        else:
            eval_started = time.perf_counter()
            eval_result = _evaluate(
                stdout_text,
                json_output_path,
                task["ground_truth"],
                args.numeric_abs_tol,
                args.numeric_rel_tol,
            )
            eval_seconds = time.perf_counter() - eval_started

        result = {
            "name": name,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "returncode": returncode,
            "success": bool(eval_result["success"] and returncode == 0),
            "errors": eval_result["errors"],
            "numeric_match": eval_result["numeric_match"],
            "numeric_details": eval_result["numeric_details"],
            "name_f1": eval_result["name_f1"],
            "ground_truth_loaded": task["ground_truth"] is not None,
            "json_output_path": json_output_path,
            "mode": "eval_only" if args.eval_only else "run_only" if args.run_only else "run_and_eval",
            "run_seconds": run_seconds,
            "eval_seconds": eval_seconds,
            "total_seconds": time.perf_counter() - started,
        }
        results.append(result)
        _write_log_files(logs_dir, task, stdout_text, result)
        print("\n=== TASK RESULT ===")
        print(json.dumps(result, indent=2))

    logs_dir.mkdir(parents=True, exist_ok=True)
    summary_json = logs_dir / f"raster_summary_{_safe_utc_ts()}.json"
    summary_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nJSON summary written to: {summary_json}")

    summary_csv = logs_dir / f"raster_summary_{_safe_utc_ts()}.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "returncode",
                "success",
                "errors",
                "numeric_match",
                "name_f1",
                "ground_truth_loaded",
                "run_seconds",
                "eval_seconds",
                "total_seconds",
                "json_output_path",
            ],
        )
        writer.writeheader()
        for r in results:
            row = dict(r)
            row["errors"] = ";".join(r.get("errors", []))
            writer.writerow({k: row.get(k) for k in writer.fieldnames})
    print(f"CSV summary written to: {summary_csv}")


if __name__ == "__main__":
    main()
