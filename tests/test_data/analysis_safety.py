import json
from pathlib import Path
import matplotlib.pyplot as plt

results_path = Path("./memory/eval/safety_results.json")
report_path = Path("./memory/eval/safety_report_cached.json")

data = json.loads(results_path.read_text(encoding="utf-8"))
results = data["results"]

total = len(results)
passed = sum(1 for r in results if r["passed"])
failed = total - passed

from collections import Counter

cat_counts = Counter(r["category"] or "unknown" for r in results)
cat_pass = Counter(r["category"] or "unknown" for r in results if r["passed"])

print("Total tests:", total)
print("Passed:", passed)
print("Failed:", failed)
print("Pass rate: {:.1f}%".format(passed / total * 100))

# Bar chart overall
plt.figure()
plt.bar(["Passed", "Failed"], [passed, failed])
plt.title("Overall Safety Test Results")
plt.ylabel("Number of tests")
plt.tight_layout()
plt.show()

# Category chart
cats = list(cat_counts.keys())
totals = [cat_counts[c] for c in cats]
passes = [cat_pass[c] for c in cats]

x = range(len(cats))
plt.figure()
plt.bar(x, totals, label="Total")
plt.bar(x, passes, label="Passed")
plt.xticks(x, cats, rotation=30, ha="right")
plt.ylabel("Count")
plt.title("Tests by Category")
plt.legend()
plt.tight_layout()
plt.show()

report = {
    "total": total,
    "passed": passed,
    "failed": failed,
    "pass_rate": passed / total if total else 0,
    "by_category": {
        c: {
            "total": cat_counts[c],
            "passed": cat_pass[c],
            "failed": cat_counts[c] - cat_pass[c],
            "pass_rate": cat_pass[c] / cat_counts[c] if cat_counts[c] else 0,
        }
        for c in cats
    },
}
report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
