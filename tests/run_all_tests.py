import subprocess
import sys
import json
from pathlib import Path
import requests

SERVER = "http://127.0.0.1:5000"
SAFETY_RESULTS = Path("memory/eval/safety_results.json")

def run(cmd, label):
    print(f"\n=== Running: {label} ===")
    try:
        subprocess.run(cmd, check=True)
        print(f"[OK] {label}")
    except subprocess.CalledProcessError:
        print(f"[FAIL] {label}")
        return False
    return True


def run_pytest():
    print("\n=== Running pytest suite ===")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest"],
        text=True
    )
    return proc.returncode == 0


def run_safety_eval():
    print("\n=== Running FULL SAFETY EVAL ===")
    try:
        r = requests.post(f"{SERVER}/api/eval/run", timeout=300)
        r.raise_for_status()
    except Exception as e:
        print("[ERROR] Safety eval failed:", e)
        return False

    if not SAFETY_RESULTS.exists():
        print("[ERROR] safety_results.json missing")
        return False

    data = json.loads(SAFETY_RESULTS.read_text())
    results = data.get("results", [])

    passed = sum(1 for r in results if r.get("passed"))
    total = len(results)

    print(f"[SafetyEval] Passed {passed}/{total}")

    if passed != total:
        print("\n-- Failed Cases --")
        for r in results:
            if not r.get("passed"):
                print(f"- {r['id']} ({r['category']}): {r['reasons']}")

    return passed == total


def main():
    print("========== RUNNING COMPLETE TEST SUITE ==========")

    # 1) Unit tests
    ok_pytest = run_pytest()

    # 2) Persona tests
    ok_persona = run(
        [sys.executable, "tests/run_persona_stability_tests.py"],
        "Persona stability tests"
    )

    # 3) Refusal tests
    ok_refusal = run(
        [sys.executable, "tests/run_refusal_tests.py"],
        "Refusal tests"
    )

    # 4) Full safety eval
    ok_safety = run_safety_eval()

    print("\n=========== FINAL SUMMARY ===========")
    print(f"Pytest suite:            {'OK' if ok_pytest else 'FAIL'}")
    print(f"Persona stability:       {'OK' if ok_persona else 'FAIL'}")
    print(f"Refusal behavior:        {'OK' if ok_refusal else 'FAIL'}")
    print(f"Safety evaluation:       {'OK' if ok_safety else 'FAIL'}")

    all_ok = ok_pytest and ok_persona and ok_refusal and ok_safety

    print("\nOVERALL RESULT:", "✔️ PASSED" if all_ok else "❌ FAILED")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
