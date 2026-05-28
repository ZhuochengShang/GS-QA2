#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
from pathlib import Path


NAME_KEYS = ("poi_name", "park_name", "lake_name", "road_name", "region_name", "name")
NUMERIC_KEYS = ("count", "area", "length", "distance", "?column?", "st_area", "st_length", "st_distance")


def latest_run_dir(result_dir: Path) -> Path:
    if (result_dir / "-predictions.json").exists():
        return result_dir
    candidates = [p for p in result_dir.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directory found under {result_dir}")
    return sorted(candidates)[-1]


def sql_from_prediction(value) -> str:
    if not isinstance(value, str):
        return ""
    return value.split("\t----- bird -----\t", 1)[0].strip()


def sql_from_history(history: list) -> str:
    for step in reversed(history):
        final_sql = step.get("final_SQL") or step.get("final_sql")
        if isinstance(final_sql, dict):
            sql = final_sql.get("PREDICTED_SQL") or final_sql.get("SQL")
            if sql:
                return str(sql).strip()
        sql = step.get("SQL")
        if sql:
            return str(sql).strip()
    return ""


def extract_sql_candidate(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"(?is)corrected query\W+(WITH\b.*?;|SELECT\b.*?;)", text)
    if match:
        return " ".join(match.group(1).split())
    match = re.search(r"(?is)```sql\s*(.*?)```", text)
    if match:
        return " ".join(match.group(1).split())
    match = re.search(r"(?s)\b(WITH\b.*?;|SELECT\b.*?;)", text)
    if match:
        return " ".join(match.group(1).split())
    return " ".join(text.split())


def final_status(history: list) -> tuple[str, str]:
    for step in reversed(history):
        final_sql = step.get("final_SQL") or step.get("final_sql")
        if isinstance(final_sql, dict):
            return str(final_sql.get("exec_res", "")), str(final_sql.get("exec_err", ""))
        if step.get("tool_name") == "execution_accuracy":
            for value in step.values():
                if isinstance(value, dict) and "exec_res" in value:
                    return str(value.get("exec_res", "")), str(value.get("exec_err", ""))
    return "", ""


def load_json(path: Path):
    return json.loads(path.read_text())


def benchmark_question_path(benchmark_root: Path, task: str, qid: int) -> Path:
    return benchmark_root / task / f"{qid:3d}" / "question.json"


def normalize_name(value) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).lower()))


def token_f1(prediction: str, truth: str) -> tuple[float, float, float]:
    pred_tokens = normalize_name(prediction).split()
    truth_tokens = normalize_name(truth).split()
    if not pred_tokens or not truth_tokens:
        return 0.0, 0.0, 0.0
    pred_counts = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    common = 0
    for token in truth_tokens:
        if pred_counts.get(token, 0) > 0:
            common += 1
            pred_counts[token] -= 1
    if common == 0:
        return 0.0, 0.0, 0.0
    precision = common / len(pred_tokens)
    recall = common / len(truth_tokens)
    return precision, recall, 2 * precision * recall / (precision + recall)


def get_name(record: dict):
    for key in NAME_KEYS:
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    if "wikipedia" in record and record["wikipedia"]:
        return str(record["wikipedia"])[3:]
    return None


def names_from_records(records: list[dict]) -> list[str]:
    names = []
    for record in records:
        if isinstance(record, dict):
            name = get_name(record)
            if name:
                names.append(name)
    return names


def first_number(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(match.group(0)) if match else None


def numbers_from_records(records: list[dict]) -> list[float]:
    values = []
    for record in records:
        if not isinstance(record, dict):
            continue
        for key, value in record.items():
            key_l = str(key).lower()
            if key_l in NUMERIC_KEYS or any(part in key_l for part in ("count", "area", "length", "distance", "angle", "azimuth")):
                num = first_number(value)
                if num is not None:
                    values.append(num)
        if not values and len(record) == 1:
            num = first_number(next(iter(record.values())))
            if num is not None:
                values.append(num)
    return values


def point_from_wkt(wkt: str):
    if not isinstance(wkt, str):
        return None
    match = re.search(r"POINT\s*\(\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*\)", wkt, re.I)
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def haversine_m(point_a, point_b) -> float:
    lon1, lat1 = point_a
    lon2, lat2 = point_b
    radius = 6371008.8
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def points_from_records(records: list[dict]) -> list[tuple[float, float]]:
    points = []
    for record in records:
        if not isinstance(record, dict):
            continue
        if "geometry" in record:
            point = point_from_wkt(record["geometry"])
            if point:
                points.append(point)
        lon = record.get("lon", record.get("longitude"))
        lat = record.get("lat", record.get("latitude"))
        if lon is not None and lat is not None:
            points.append((float(lon), float(lat)))
    return points


def relative_error(pred: float, true: float):
    if true == 0:
        return 0.0 if pred == 0 else float("inf")
    return abs(pred - true) / abs(true)


def angle_error_degrees(pred: float, true: float) -> float:
    diff = abs(pred - true) % 360
    return min(diff, 360 - diff)


def evaluate_question(gold: dict, predicted: dict, args) -> dict:
    qtype = gold.get("type", "")
    gold_answers = gold.get("answers", [])
    pred_answers = predicted.get("answers", [])
    row = {
        "type": qtype,
        "attempted": bool(pred_answers),
        "passed": False,
        "metric": "",
        "score": "",
        "prediction": "",
        "truth": "",
    }

    if "name" in qtype or "multi_source" in qtype:
        pred_names = names_from_records(pred_answers)
        true_names = [name for name in (get_name(ans) for ans in gold_answers if isinstance(ans, dict)) if name]
        pred_text = "\n".join(pred_names)
        true_text = "\n".join(true_names)
        precision, recall, f1 = token_f1(pred_text, true_text)
        row.update({
            "attempted": bool(pred_names),
            "passed": f1 >= args.name_f1_tolerance,
            "metric": "name_f1",
            "score": round(f1, 6),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "prediction": pred_text,
            "truth": true_text,
        })
        return row

    if "loc" in qtype:
        pred_points = points_from_records(pred_answers)
        true_points = [point_from_wkt(ans.get("geometry")) for ans in gold_answers if isinstance(ans, dict)]
        true_points = [point for point in true_points if point]
        distances = [haversine_m(pred, true) for pred in pred_points for true in true_points]
        best = min(distances) if distances else float("inf")
        row.update({
            "attempted": bool(pred_points),
            "passed": best <= args.location_tolerance_m,
            "metric": "location_error_m",
            "score": round(best, 3) if math.isfinite(best) else "",
            "prediction": json.dumps(pred_points),
            "truth": json.dumps(true_points),
        })
        return row

    if "angle" in qtype:
        preds = numbers_from_records(pred_answers)
        truths = [first_number(ans.get("angle")) for ans in gold_answers if isinstance(ans, dict)]
        truths = [value for value in truths if value is not None]
        errors = [angle_error_degrees(pred, true) for pred in preds for true in truths]
        best = min(errors) if errors else float("inf")
        row.update({
            "attempted": bool(preds),
            "passed": best <= args.angle_tolerance_deg,
            "metric": "angle_error_deg",
            "score": round(best, 6) if math.isfinite(best) else "",
            "prediction": json.dumps(preds),
            "truth": json.dumps(truths),
        })
        return row

    numeric_key = next((key for key in ("count", "area", "length", "distance") if key in qtype), "")
    if numeric_key:
        preds = numbers_from_records(pred_answers)
        truths = [first_number(ans.get(numeric_key)) for ans in gold_answers if isinstance(ans, dict)]
        truths = [value for value in truths if value is not None]
        pred = preds[0] if preds else None
        true = truths[0] if truths else None
        err = relative_error(pred, true) if pred is not None and true is not None else float("inf")
        abs_err = abs(pred - true) if pred is not None and true is not None else float("inf")
        passed = err <= args.numeric_relative_tolerance
        if numeric_key == "count" and math.isfinite(abs_err):
            passed = passed or abs_err <= args.count_absolute_tolerance
        row.update({
            "attempted": pred is not None,
            "passed": passed,
            "metric": f"{numeric_key}_relative_error",
            "score": round(err, 6) if math.isfinite(err) else "",
            "absolute_error": round(abs_err, 6) if math.isfinite(abs_err) else "",
            "prediction": pred if pred is not None else "",
            "truth": true if true is not None else "",
        })
        return row

    row["metric"] = "unsupported_type"
    return row


def export_generated_sql(chess_root: Path, benchmark_root: Path, output_path: Path):
    rows = []
    for result_dir in sorted(chess_root.glob("gsqa_T*_postgis"), key=lambda p: int(re.search(r"T(\d+)", p.name).group(1))):
        task = re.search(r"(T\d+)", result_dir.name).group(1)
        run_dir = latest_run_dir(result_dir)
        predictions_path = run_dir / "-predictions.json"
        predictions = load_json(predictions_path) if predictions_path.exists() else {}
        history_paths = [p for p in run_dir.glob("*_*.json") if p.name.split("_", 1)[0].isdigit()]
        for path in sorted(history_paths, key=lambda p: int(p.name.split("_", 1)[0])):
            qid = int(path.name.split("_", 1)[0])
            db_id = path.stem.split("_", 1)[1]
            history = load_json(path)
            sql_raw = sql_from_history(history) or sql_from_prediction(predictions.get(str(qid)))
            exec_res, exec_err = final_status(history)
            gold_path = benchmark_question_path(benchmark_root, task, qid)
            gold = load_json(gold_path) if gold_path.exists() else {}
            rows.append({
                "task": task,
                "question_id": qid,
                "db_id": db_id,
                "type": gold.get("type", ""),
                "question": gold.get("question", ""),
                "sql": extract_sql_candidate(sql_raw),
                "sql_raw": sql_raw,
                "final_exec_res": exec_res,
                "final_exec_err": exec_err,
            })
    write_table(output_path, rows)
    return rows


def evaluate_results(chess_root: Path, benchmark_root: Path, output_path: Path, args):
    rows = []
    for result_dir in sorted(chess_root.glob("T*_sql_results"), key=lambda p: int(p.name.split("_", 1)[0][1:])):
        task = result_dir.name.split("_", 1)[0]
        for pred_path in sorted(result_dir.glob("*/question.json"), key=lambda p: int(p.parent.name.strip())):
            qid = int(pred_path.parent.name.strip())
            gold_path = benchmark_question_path(benchmark_root, task, qid)
            if not gold_path.exists():
                continue
            gold = load_json(gold_path)
            pred = load_json(pred_path)
            row = evaluate_question(gold, pred, args)
            row.update({
                "task": task,
                "question_id": qid,
                "question": gold.get("question", ""),
                "sql": pred.get("sql", ""),
                "final_exec_res": pred.get("metadata", {}).get("final_exec_res", ""),
                "final_exec_err": pred.get("metadata", {}).get("final_exec_err", ""),
            })
            rows.append(row)
    rows.sort(key=lambda row: (int(row["task"][1:]), int(row["question_id"])))
    write_table(output_path, rows)
    summary_path = output_path.with_name(output_path.stem + "_summary.csv")
    summary_rows = summarize(rows)
    write_table(summary_path, summary_rows)
    return rows, summary_rows


def summarize(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        key = (row["task"], row["type"])
        groups.setdefault(key, []).append(row)
    summary = []
    for (task, qtype), items in sorted(groups.items(), key=lambda kv: (int(kv[0][0][1:]), kv[0][1])):
        attempted = sum(1 for row in items if row.get("attempted"))
        passed = sum(1 for row in items if row.get("passed"))
        scores = [float(row["score"]) for row in items if row.get("score") not in ("", None)]
        summary.append({
            "task": task,
            "type": qtype,
            "n": len(items),
            "attempted": attempted,
            "passed": passed,
            "attempted_rate": round(attempted / len(items), 6) if items else 0,
            "pass_rate": round(passed / len(items), 6) if items else 0,
            "mean_score": round(sum(scores) / len(scores), 6) if scores else "",
        })
    if rows:
        attempted = sum(1 for row in rows if row.get("attempted"))
        passed = sum(1 for row in rows if row.get("passed"))
        summary.append({
            "task": "ALL",
            "type": "ALL",
            "n": len(rows),
            "attempted": attempted,
            "passed": passed,
            "attempted_rate": round(attempted / len(rows), 6),
            "pass_rate": round(passed / len(rows), 6),
            "mean_score": "",
        })
    return summary


def write_table(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = ["task", "question_id", "type", "attempted", "passed", "metric", "score", "absolute_error", "precision", "recall", "prediction", "truth", "question", "sql", "final_exec_res", "final_exec_err"]
    fieldnames = [key for key in preferred if key in fieldnames] + [key for key in fieldnames if key not in preferred]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Export and evaluate generated CHESS SQL results with tolerant scoring.")
    parser.add_argument("--chess-root", type=Path, default=Path("CHESS/CHESS_IR_SS_CG_GEMINI"))
    parser.add_argument("--benchmark-root", type=Path, default=Path("benchmark"))
    parser.add_argument("--sql-output", type=Path, default=Path("CHESS/CHESS_IR_SS_CG_GEMINI/chess_generated_sql.csv"))
    parser.add_argument("--eval-output", type=Path, default=Path("CHESS/CHESS_IR_SS_CG_GEMINI/chess_sql_tolerance_eval.csv"))
    parser.add_argument("--export-sql", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--numeric-relative-tolerance", type=float, default=0.05)
    parser.add_argument("--count-absolute-tolerance", type=float, default=0.0)
    parser.add_argument("--name-f1-tolerance", type=float, default=0.5)
    parser.add_argument("--location-tolerance-m", type=float, default=500.0)
    parser.add_argument("--angle-tolerance-deg", type=float, default=22.5)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.export_sql and not args.evaluate:
        args.export_sql = True
        args.evaluate = True
    if args.export_sql:
        sql_rows = export_generated_sql(args.chess_root, args.benchmark_root, args.sql_output)
        print(f"exported_sql: {len(sql_rows)} rows -> {args.sql_output}")
    if args.evaluate:
        eval_rows, summary_rows = evaluate_results(args.chess_root, args.benchmark_root, args.eval_output, args)
        print(f"evaluated: {len(eval_rows)} rows -> {args.eval_output}")
        print(f"summary: {args.eval_output.with_name(args.eval_output.stem + '_summary.csv')}")
        if summary_rows:
            print(f"overall_pass_rate: {summary_rows[-1]['pass_rate']}")


if __name__ == "__main__":
    main()
