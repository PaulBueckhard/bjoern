import json, requests
from pathlib import Path

SERVER = "http://127.0.0.1:5000"
PROMPTS = Path("tests/prompts/safety_prompts.json")
OUT = Path("memory/eval/safety_results.json")

def main():
    prompts = json.loads(PROMPTS.read_text())

    print(f"[SafetyEval] Running {len(prompts)} prompts…")

    # Run evaluation
    r = requests.post(f"{SERVER}/api/eval/run")
    r.raise_for_status()

    # Load results
    data = json.loads(OUT.read_text())
    results = data["results"]

    safe = sum(1 for x in results if x["safety"]["safe"])
    unsafe = len(results) - safe

    print("\n===== SAFETY SUMMARY =====")
    print(f"Total: {len(results)}")
    print(f"Safe: {safe}")
    print(f"Unsafe: {unsafe}")

    if unsafe:
        print("\nExamples of UNSAFE responses:")
        for x in results:
            if not x["safety"]["safe"]:
                print("- Prompt:", x["prompt"])
                print("  Reply :", x["reply"])
                print("  Hits  :", x["safety"])
                print()
                break

if __name__ == "__main__":
    main()
