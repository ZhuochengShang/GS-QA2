import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
from statistics import mean, median

def _safe_utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _collect_results(logs_dir: Path):
    results = []
    for path in sorted(logs_dir.glob("*.result.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict) and "name" in data:
            data["_path"] = str(path)
            results.append(data)
    return results


def _num(x):
    return isinstance(x, (int, float))


def main():
    parser = argparse.ArgumentParser(description="Summarize all *.result.json logs")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--out", default=None)
    parser.add_argument("--name-f1-threshold", type=float, default=0.5)
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    results = _collect_results(logs_dir)

    total = len(results)
    success = [r for r in results if r.get("success") is True]
    name_f1_vals = [r.get("name_f1") for r in results if _num(r.get("name_f1"))]
    geom_vals = [r.get("geom_dist_m") for r in results if _num(r.get("geom_dist_m"))]

    tokens_total = 0
    prompt_tokens = 0
    completion_tokens = 0
    cost_total = 0.0

    by_model = {}
    for r in results:
        usage = r.get("usage") or {}
        cost = r.get("cost_estimate") or {}
        if isinstance(usage, dict):
            prompt_tokens += usage.get("prompt_tokens", 0) or 0
            completion_tokens += usage.get("completion_tokens", 0) or 0
            tokens_total += usage.get("total_tokens", 0) or 0
            for model, u in (usage.get("by_model") or {}).items():
                m = by_model.setdefault(model, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "estimated_cost_usd": 0.0})
                m["prompt_tokens"] += u.get("prompt_tokens", 0) or 0
                m["completion_tokens"] += u.get("completion_tokens", 0) or 0
                m["total_tokens"] += u.get("total_tokens", 0) or 0
        if isinstance(cost, dict):
            for model, c in cost.items():
                est = (c.get("estimated_cost_usd") if isinstance(c, dict) else 0) or 0
                cost_total += est
                m = by_model.setdefault(model, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "estimated_cost_usd": 0.0})
                m["estimated_cost_usd"] += est

    name_match_count = sum(1 for r in results if _num(r.get("name_f1")) and r.get("name_f1") >= args.name_f1_threshold)

    summary = {
        "logs_dir": str(logs_dir),
        "total": total,
        "success_count": len(success),
        "success_rate": (len(success) / total) if total else 0.0,
        "name_f1_count": len(name_f1_vals),
        "name_f1_mean": mean(name_f1_vals) if name_f1_vals else None,
        "name_f1_median": median(name_f1_vals) if name_f1_vals else None,
        "name_match_threshold": args.name_f1_threshold,
        "name_match_count": name_match_count,
        "name_match_rate": (name_match_count / total) if total else 0.0,
        "geom_dist_count": len(geom_vals),
        "geom_dist_mean": mean(geom_vals) if geom_vals else None,
        "geom_dist_median": median(geom_vals) if geom_vals else None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": tokens_total,
        "estimated_cost_usd": cost_total,
        "by_model": by_model,
        "results": results,
    }

    out = args.out
    if not out:
        out = logs_dir / f"summary_all_{_safe_utc_ts()}.json"
    out = Path(out)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
