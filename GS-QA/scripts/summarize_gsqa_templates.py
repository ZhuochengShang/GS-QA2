import json
from pathlib import Path

for i in range(1, 29):
    files = sorted(Path(f"benchmark/T{i}").glob("*/question.json"))
    obj = json.loads(files[0].read_text()) if files else {}
    answers = obj.get("answers") or []
    keys = sorted({key for answer in answers if isinstance(answer, dict) for key in answer})
    print(
        f"T{i}: type={obj.get('type')!r} "
        f"answer_keys={keys} "
        f"question={obj.get('question', '')[:150]!r}"
    )
