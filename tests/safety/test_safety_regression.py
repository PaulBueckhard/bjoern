import json
from pathlib import Path

EVAL_RESULTS = Path("memory/eval/safety_results.json")

def test_regression_no_unsafe_outputs():
    if not EVAL_RESULTS.exists():
        return  # nothing to test yet

    data = json.loads(EVAL_RESULTS.read_text())
    for item in data:
        assert item["safety"]["safe"], f"Unsafe reply detected for prompt: {item['prompt']}"
