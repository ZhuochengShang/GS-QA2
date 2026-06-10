import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/zshan011/SpatialAnalysisAgent-master/SpatialAnalysisAgent")
DATA_ROOT = "/local_data/scratch/zshan011/osm/osm_extract"
CODE_DIR = REPO / "workspace/task_artifacts/generated_code"
OUT_DIR = REPO / "workspace/direct_generated_exec/latest_check"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TIMEOUT = 360


def patch_source(src: str) -> str:
    src = src.replace("/home/zshan011/OsmData", DATA_ROOT)
    src = src.replace("./OsmData", DATA_ROOT)
    src = src.replace('"OsmData"', json.dumps(DATA_ROOT))
    src = src.replace("'OsmData'", repr(DATA_ROOT))

    tail = "\n".join(src.splitlines()[-8:])
    if re.search(r"^\s*def\s+main\s*\(\s*data_root\s*\)\s*:", src, re.M) and "main(" not in tail:
        src += f"\n\nif __name__ == '__main__':\n    print(json.dumps(main({DATA_ROOT!r})))\n"
    elif re.search(r"^\s*def\s+main\s*\(\s*data_root\s*=", src, re.M) and "main(" not in tail:
        src += f"\n\nif __name__ == '__main__':\n    print(json.dumps(main({DATA_ROOT!r})))\n"
    elif re.search(r"^\s*def\s+solve\s*\(", src, re.M) and "solve()" not in tail:
        src += "\n\nif __name__ == '__main__':\n    solve()\n"
    return src


def last_json(stdout: str):
    for line in reversed([line.strip() for line in stdout.splitlines() if line.strip()]):
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


names = sys.argv[1:] or [p.stem for p in sorted(CODE_DIR.glob("*.py"), key=lambda p: p.stat().st_mtime, reverse=True)[:5]]
env = os.environ.copy()
env["PROJ_LIB"] = "/home/zshan011/anaconda3/envs/spatialagent/share/proj"
env["GDAL_DATA"] = "/home/zshan011/anaconda3/envs/spatialagent/share/gdal"
env["GS_DATA_ROOT"] = DATA_ROOT
env["SPATIALAGENT_DATA_PATH"] = DATA_ROOT

for name in names:
    src_path = CODE_DIR / f"{name}.py"
    if not src_path.exists():
        print(json.dumps({"name": name, "missing": True}))
        continue
    patched = OUT_DIR / f"{name}.patched.py"
    patched.write_text(patch_source(src_path.read_text(encoding="utf-8", errors="replace")), encoding="utf-8")
    started = time.perf_counter()
    proc = subprocess.Popen(
        ["/home/zshan011/anaconda3/envs/spatialagent/bin/python", str(patched)],
        cwd=str(REPO),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=TIMEOUT)
        timed_out = False
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        timed_out = True
        rc = 124
    elapsed = time.perf_counter() - started
    (OUT_DIR / f"{name}.stdout.txt").write_text(stdout or "", encoding="utf-8")
    (OUT_DIR / f"{name}.stderr.txt").write_text(stderr or "", encoding="utf-8")
    parsed = last_json(stdout or "")
    answers = parsed.get("answers") if isinstance(parsed, dict) else None
    print(
        json.dumps(
            {
                "name": name,
                "returncode": rc,
                "timed_out": timed_out,
                "seconds": elapsed,
                "stdout_bytes": len(stdout or ""),
                "stderr_bytes": len(stderr or ""),
                "parsed_json": parsed is not None,
                "answer_count": len(answers) if isinstance(answers, list) else None,
                "has_answer": bool(answers),
                "stdout": str(OUT_DIR / f"{name}.stdout.txt"),
                "stderr": str(OUT_DIR / f"{name}.stderr.txt"),
                "patched": str(patched),
            }
        ),
        flush=True,
    )
