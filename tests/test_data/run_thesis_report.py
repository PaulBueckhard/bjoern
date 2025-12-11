import subprocess
import sys
import json
import datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent.parent
REPORT_DIR = ROOT / "memory" / "eval"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = REPORT_DIR / "thesis_test_report.md"


def run_cmd(cmd):
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return {
        "cmd": " ".join(str(c) for c in cmd),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def parse_pytest_summary(stdout):
    total_collected = None
    passed = failed = skipped = xfailed = xpassed = None
    lines = stdout.splitlines()
    for line in lines:
        if "collected" in line and "item" in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "collected" and i + 1 < len(parts):
                    try:
                        total_collected = int(parts[i + 1])
                    except ValueError:
                        pass
        if " passed" in line or " failed" in line or " skipped" in line:
            tokens = line.replace(",", "").split()
            i = 0
            while i + 1 < len(tokens):
                try:
                    n = int(tokens[i])
                except ValueError:
                    i += 1
                    continue
                label = tokens[i + 1]
                if label.startswith("passed"):
                    passed = (passed or 0) + n
                elif label.startswith("failed"):
                    failed = (failed or 0) + n
                elif label.startswith("skipped"):
                    skipped = (skipped or 0) + n
                elif label.startswith("xfailed"):
                    xfailed = (xfailed or 0) + n
                elif label.startswith("xpassed"):
                    xpassed = (xpassed or 0) + n
                i += 2
    return {
        "collected": total_collected,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "xfailed": xfailed,
        "xpassed": xpassed,
    }


def parse_failures_line(stdout):
    failures = total = None
    for line in stdout.splitlines():
        if "Failures:" in line:
            parts = line.split("Failures:")[-1].strip()
            if "/" in parts:
                left, right = parts.split("/", 1)
                try:
                    failures = int(left.strip())
                    total = int(right.strip())
                except ValueError:
                    failures = total = None
    return {"failures": failures, "total": total}


def run_pytest():
    return run_cmd([sys.executable, "-m", "pytest"])


def run_persona_tests():
    return run_cmd([sys.executable, "tests/run_persona_stability_tests.py"])


def run_refusal_tests():
    return run_cmd([sys.executable, "tests/run_refusal_tests.py"])


def run_safety_eval():
    url_run = "http://127.0.0.1:5000/api/eval/run"
    url_report = "http://127.0.0.1:5000/api/eval/report"
    result = {"ok": False, "error": None, "report": None}
    try:
        r = requests.post(url_run, timeout=300)
    except Exception as e:
        result["error"] = f"POST {url_run} failed: {e}"
        return result
    if not r.ok:
        result["error"] = f"POST {url_run} returned status {r.status_code}"
        return result
    try:
        r2 = requests.get(url_report, timeout=60)
    except Exception as e:
        result["error"] = f"GET {url_report} failed: {e}"
        return result
    if not r2.ok:
        try:
            data = r2.json()
        except Exception:
            data = None
        result["error"] = f"GET {url_report} returned status {r2.status_code} body={data}"
        return result
    try:
        data = r2.json()
    except Exception as e:
        result["error"] = f"Decoding report JSON failed: {e}"
        return result
    result["ok"] = True
    result["report"] = data
    return result


def tail(text, max_lines=40):
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[-max_lines:])


def main():
    generated_at = datetime.datetime.now().isoformat()
    python_version = sys.version.replace("\n", " ")
    sections = []

    pytest_res = run_pytest()
    pytest_summary = parse_pytest_summary(pytest_res["stdout"])
    sec = []
    sec.append("# Björn Test Suite Report")
    sec.append("")
    sec.append(f"Generated: `{generated_at}`")
    sec.append(f"Python: `{python_version}`")
    sec.append("")
    sec.append("## 1. Pytest test suite")
    sec.append(f"- Command: `{pytest_res['cmd']}`")
    sec.append(f"- Exit code: {pytest_res['returncode']}")
    if pytest_summary["collected"] is not None:
        sec.append(f"- Collected tests: {pytest_summary['collected']}")
    if pytest_summary["passed"] is not None:
        sec.append(f"- Passed: {pytest_summary['passed']}")
    if pytest_summary["failed"] is not None:
        sec.append(f"- Failed: {pytest_summary['failed']}")
    if pytest_summary["skipped"] is not None:
        sec.append(f"- Skipped: {pytest_summary['skipped']}")
    sec.append("")
    sec.append("### Pytest output (tail)")
    sec.append("")
    sec.append("```text")
    sec.append(tail(pytest_res["stdout"]))
    if pytest_res["stderr"]:
        sec.append("")
        sec.append("--- stderr ---")
        sec.append(tail(pytest_res["stderr"]))
    sec.append("```")
    sections.append("\n".join(sec))

    persona_res = run_persona_tests()
    persona_stats = parse_failures_line(persona_res["stdout"])
    sec = []
    sec.append("## 2. Persona stability tests")
    sec.append(f"- Command: `{persona_res['cmd']}`")
    sec.append(f"- Exit code: {persona_res['returncode']}")
    if persona_stats["total"] is not None:
        sec.append(f"- Total prompts: {persona_stats['total']}")
        sec.append(f"- Failures: {persona_stats['failures']}")
    sec.append("")
    sec.append("```text")
    sec.append(tail(persona_res["stdout"]))
    if persona_res["stderr"]:
        sec.append("")
        sec.append("--- stderr ---")
        sec.append(tail(persona_res["stderr"]))
    sec.append("```")
    sections.append("\n".join(sec))

    refusal_res = run_refusal_tests()
    refusal_stats = parse_failures_line(refusal_res["stdout"])
    sec = []
    sec.append("## 3. Refusal behavior tests")
    sec.append(f"- Command: `{refusal_res['cmd']}`")
    sec.append(f"- Exit code: {refusal_res['returncode']}")
    if refusal_stats["total"] is not None:
        sec.append(f"- Total prompts: {refusal_stats['total']}")
        sec.append(f"- Failures: {refusal_stats['failures']}")
    sec.append("")
    sec.append("```text")
    sec.append(tail(refusal_res["stdout"]))
    if refusal_res["stderr"]:
        sec.append("")
        sec.append("--- stderr ---")
        sec.append(tail(refusal_res["stderr"]))
    sec.append("```")
    sections.append("\n".join(sec))

    safety_res = run_safety_eval()
    sec = []
    sec.append("## 4. Full safety evaluation")
    sec.append("- Endpoint: `POST /api/eval/run` and `GET /api/eval/report`")
    if safety_res["ok"]:
        rep = safety_res["report"] or {}
        summary = rep.get("summary") or {}
        by_cat = rep.get("by_category") or {}
        sec.append("- Status: OK")
        sec.append(f"- Total tests: {summary.get('total_tests')}")
        sec.append(f"- Passed: {summary.get('passed')}")
        sec.append(f"- Failed: {summary.get('failed')}")
        pr = summary.get("pass_rate")
        if isinstance(pr, (int, float)):
            sec.append(f"- Pass rate: {pr*100:.1f}%")
        if by_cat:
            sec.append("")
            sec.append("### Results by category")
            for cat, vals in sorted(by_cat.items()):
                t = vals.get("total")
                p = vals.get("passed")
                f = vals.get("failed")
                prc = None
                if isinstance(t, int) and t:
                    prc = p / t * 100 if isinstance(p, int) else None
                if prc is not None:
                    sec.append(f"- {cat}: {p}/{t} passed ({prc:.1f}%)")
                else:
                    sec.append(f"- {cat}: {p}/{t} passed")
        examples = rep.get("unsafe_examples") or []
        if examples:
            sec.append("")
            sec.append("### Example unsafe cases")
            for ex in examples[:5]:
                sec.append("")
                sec.append(f"- `{ex.get('id')}` ({ex.get('category')}): {ex.get('reasons')}")
    else:
        sec.append("- Status: ERROR")
        sec.append(f"- Error: {safety_res['error']}")
    sections.append("\n".join(sec))

    report = "\n\n".join(sections)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Wrote report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
