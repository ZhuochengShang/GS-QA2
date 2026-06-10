import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/zshan011/SpatialAnalysisAgent-master/SpatialAnalysisAgent")
CODE_DIR = REPO / "workspace/task_artifacts/generated_code"
OLD_OUTPUT_DIR = REPO / "workspace/task_artifacts/outputs"
OUT_DIR = REPO / "workspace/direct_generated_exec"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PATCHED_CODE_DIR = OUT_DIR / "patched_code"
PATCHED_CODE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = OUT_DIR / "direct_exec_results.jsonl"
TIMEOUT_SECONDS = int(os.environ.get("DIRECT_EXEC_TIMEOUT", os.environ.get("SPATIALAGENT_TASK_TIMEOUT", "360")))
PYTHON = os.environ.get("DIRECT_EXEC_PYTHON", sys.executable)
DATA_ROOT = os.environ.get("DIRECT_EXEC_DATA_ROOT", "/local_data/scratch/zshan011/osm/osm_extract")


def has_main_guard(text: str) -> bool:
    return "if __name__" in text and "__main__" in text


def already_done() -> set[str]:
    done = set()
    if RESULTS_PATH.exists():
        for line in RESULTS_PATH.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["name"])
            except Exception:
                pass
    return done


def last_json(stdout: str):
    for line in reversed([x.strip() for x in stdout.splitlines() if x.strip()]):
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


done = already_done()
candidates = []
for py in sorted(CODE_DIR.glob("*.py")):
    name = py.stem
    if name in done:
        continue
    old_out = OLD_OUTPUT_DIR / f"{name}.output.txt"
    if not old_out.exists() or old_out.stat().st_size != 0:
        continue
    try:
        text = py.read_text(encoding="utf-8", errors="replace")
    except Exception:
        continue
    if has_main_guard(text):
        candidates.append(py)

print(
    json.dumps(
        {
            "event": "start",
            "pending": len(candidates),
            "timeout_seconds": TIMEOUT_SECONDS,
            "results_path": str(RESULTS_PATH),
        }
    ),
    flush=True,
)

env = os.environ.copy()
env["GS_DATA_ROOT"] = DATA_ROOT
env["SPATIALAGENT_DATA_PATH"] = DATA_ROOT

with RESULTS_PATH.open("a", encoding="utf-8") as rf:
    for idx, py in enumerate(candidates, 1):
        name = py.stem
        patched_py = PATCHED_CODE_DIR / py.name
        src = py.read_text(encoding="utf-8", errors="replace")
        src = src.replace("/home/zshan011/OsmData", DATA_ROOT)
        patched_py.write_text(src, encoding="utf-8")
        started = time.perf_counter()
        stdout = ""
        stderr = ""
        timed_out = False
        try:
            proc = subprocess.Popen(
                [PYTHON, str(patched_py)],
                cwd=str(REPO),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=TIMEOUT_SECONDS)
                returncode = proc.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
                proc.kill()
                stdout, stderr = proc.communicate()
                returncode = 124
        except Exception as e:
            returncode = 999
            stderr = repr(e)

        seconds = time.perf_counter() - started
        parsed = last_json(stdout or "")
        answers = parsed.get("answers") if isinstance(parsed, dict) else None
        answer_count = len(answers) if isinstance(answers, list) else None
        (OUT_DIR / f"{name}.stdout.txt").write_text(stdout or "", encoding="utf-8")
        (OUT_DIR / f"{name}.stderr.txt").write_text(stderr or "", encoding="utf-8")
        row = {
            "name": name,
            "returncode": returncode,
            "timed_out": timed_out,
            "exec_seconds": seconds,
            "stdout_bytes": len(stdout or ""),
            "stderr_bytes": len(stderr or ""),
            "parsed_json": parsed is not None,
            "answer_count": answer_count,
            "has_answer": bool(answer_count),
            "stdout_path": str(OUT_DIR / f"{name}.stdout.txt"),
            "stderr_path": str(OUT_DIR / f"{name}.stderr.txt"),
        }
        rf.write(json.dumps(row) + "\n")
        rf.flush()
        print(json.dumps({"event": "done", "idx": idx, "total": len(candidates), **row}), flush=True)
