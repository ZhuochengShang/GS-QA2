import json
import os
import re
import subprocess
from datetime import datetime, timezone

FAIL_PHRASE = "Failed to execute and debug the code within 5 times."


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


def evaluate(stdout_text, expected_patterns, expected_paths):
    errors = []
    if FAIL_PHRASE in stdout_text:
        errors.append("agent_failed_after_5_retries")

    missing_patterns = []
    for pat in expected_patterns:
        if not re.search(pat, stdout_text, re.IGNORECASE):
            missing_patterns.append(pat)

    missing_paths = []
    for p in expected_paths:
        if not os.path.exists(p):
            missing_paths.append(p)

    success = not errors and not missing_patterns and not missing_paths
    return success, errors, missing_patterns, missing_paths


def main():
    tasks = [
        {
            "name": "ocean_facing_park",
            "env": {
                "HOME": os.environ.get("HOME", ""),
                "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
                "QGIS_PREFIX_PATH": "/Applications/QGIS.app",
                "PROJ_LIB": "/Applications/QGIS.app/Contents/Resources/qgis/proj",
                "GDAL_DATA": "/Applications/QGIS.app/Contents/Resources/gdal",
                "DYLD_LIBRARY_PATH": "/Applications/QGIS.app/Contents/Frameworks:/Applications/QGIS.app/Contents/MacOS",
                "QT_QPA_PLATFORM": "offscreen",
                "PYTHONPATH": "/Applications/QGIS.app/Contents/Resources/python:/Applications/QGIS.app/Contents/Resources/python/plugins:/Applications/QGIS.app/Contents/Resources/qgis/python",
            },
            "cmd": [
                os.environ.get("SPATIAL_AGENT_PYTHON", "python3"),
                os.environ.get("SPATIAL_AGENT_HEADLESS", "SpatialAnalysisAgent_headless.py"),
                "--task",
                "You have OSM park data and COP30 elevation data. Find one park that: The field uname is not empty. The park has a potential ocean-facing view. Return: Park name, Shapefile to workspace, Elevation context (park vs. ocean-facing view terrain), Brief justification of why it has the ocean-facing view.",
                "--data-path",
                os.environ.get("SPATIAL_AGENT_DATA_PATH", "/path/to/COP30.tif;/path/to/osm_parks.geojson"),
                "--model",
                "gpt-4o",
                "--workspace",
                os.environ.get("SPATIAL_AGENT_WORKSPACE", "workspace"),
                "--review",
                "true",
                "--reasoning-effort",
                "medium",
            ],
            "expected_patterns": [
                r"Parks with ocean-facing view potential",
            ],
            "expected_paths": [
            ],
        },
    ]

    results = []
    for task in tasks:
        print(f"\n=== Running task: {task['name']} ===")
        returncode, stdout_text = run_task(task)

        success, errors, missing_patterns, missing_paths = evaluate(
            stdout_text,
            task.get("expected_patterns", []),
            task.get("expected_paths", []),
        )

        result = {
            "name": task["name"],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "returncode": returncode,
            "success": success and returncode == 0,
            "errors": errors,
            "missing_patterns": missing_patterns,
            "missing_paths": missing_paths,
        }
        results.append(result)

        print("\n=== TASK RESULT ===")
        print(json.dumps(result, indent=2))

    print("\n=== ALL RESULTS ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
