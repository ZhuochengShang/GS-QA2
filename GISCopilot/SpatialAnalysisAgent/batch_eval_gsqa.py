import argparse
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Tuple


def _safe_utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _run_auto_eval(python_bin: str, auto_eval_path: str, tasks_json: str) -> str:
    cmd = [python_bin, auto_eval_path, "--tasks-json", tasks_json, "--all"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    stdout = proc.stdout or ""
    m = re.search(r"JSON summary written to: (.+)", stdout)
    if not m:
        raise RuntimeError(f"Failed to find summary path in output for {tasks_json}\n{stdout}")
    return m.group(1).strip()


def _summarize_results(results: List[Dict[str, Any]], name_f1_threshold: float) -> Dict[str, Any]:
    total = len(results)
    success_count = sum(1 for r in results if r.get("success") is True)
    name_f1_present = sum(1 for r in results if r.get("name_f1") is not None)
    name_match_count = 0
    for r in results:
        f1 = r.get("name_f1")
        if isinstance(f1, (int, float)) and f1 >= name_f1_threshold:
            name_match_count += 1
    success_rate = (success_count / total) if total else 0.0
    name_match_rate = (name_match_count / total) if total else 0.0
    return {
        "total": total,
        "success_count": success_count,
        "success_rate": success_rate,
        "name_f1_present": name_f1_present,
        "name_match_count": name_match_count,
        "name_match_rate": name_match_rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch runner for GS-QA auto_eval_GSQA.py")
    parser.add_argument("--questions-dir", required=True, help="Directory containing JSONL files")
    parser.add_argument(
        "--auto-eval",
        default="auto_eval_GSQA.py",
        help="Path to auto_eval_GSQA.py",
    )
    parser.add_argument(
        "--python-bin",
        default="python3",
        help="Python interpreter to use for running auto_eval_GSQA.py",
    )
    parser.add_argument(
        "--name-f1-threshold",
        type=float,
        default=0.5,
        help="Threshold for name match rate",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON path for batch summary (default: logs/batch_summary_<ts>.json)",
    )
    args = parser.parse_args()

    questions_dir = Path(args.questions_dir)
    if not questions_dir.exists():
        raise SystemExit(f"questions-dir not found: {questions_dir}")

    jsonl_files = sorted(p for p in questions_dir.rglob("*.jsonl"))
    if not jsonl_files:
        raise SystemExit(f"no .jsonl files found under {questions_dir}")

    per_file = []
    overall_totals = {
        "total": 0,
        "success_count": 0,
        "name_f1_present": 0,
        "name_match_count": 0,
    }

    for jsonl_path in jsonl_files:
        summary_path = _run_auto_eval(args.python_bin, args.auto_eval, str(jsonl_path))
        with open(summary_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        summary = _summarize_results(results, args.name_f1_threshold)
        per_file.append(
            {
                "file": str(jsonl_path),
                **summary,
                "summary_path": summary_path,
            }
        )
        overall_totals["total"] += summary["total"]
        overall_totals["success_count"] += summary["success_count"]
        overall_totals["name_f1_present"] += summary["name_f1_present"]
        overall_totals["name_match_count"] += summary["name_match_count"]

    overall = {
        "total": overall_totals["total"],
        "success_count": overall_totals["success_count"],
        "success_rate": (overall_totals["success_count"] / overall_totals["total"]) if overall_totals["total"] else 0.0,
        "name_f1_present": overall_totals["name_f1_present"],
        "name_match_count": overall_totals["name_match_count"],
        "name_match_rate": (overall_totals["name_match_count"] / overall_totals["total"]) if overall_totals["total"] else 0.0,
        "name_f1_threshold": args.name_f1_threshold,
    }

    out_path = args.out
    if not out_path:
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(logs_dir / f"batch_summary_{_safe_utc_ts()}.json")

    payload = {
        "questions_dir": str(questions_dir),
        "name_f1_threshold": args.name_f1_threshold,
        "per_file": per_file,
        "overall": overall,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(out_path)


if __name__ == "__main__":
    main()
