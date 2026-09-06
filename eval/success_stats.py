#!/usr/bin/env python3
"""Turn raw per-rollout logs into success rates with Wilson confidence intervals."""
import csv
import math
import os
from collections import defaultdict

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "results.csv")

def wilson(k, n, z=1.96):
    """Return (point_estimate, ci_low, ci_high) for k successes in n trials."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, center - margin), min(1.0, center + margin)

def load_and_aggregate(path=RESULTS):
    groups = defaultdict(lambda: [0, 0])  # (policy, trained_on) -> [successes, n]
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["policy"], row["trained_on"])
            groups[key][0] += int(row["success"])
            groups[key][1] += 1

    out = []
    for (policy, trained_on), (k, n) in groups.items():
        rate, lo, hi = wilson(k, n)
        out.append({
            "policy": policy, "trained_on": trained_on,
            "n": n, "successes": k,
            "rate": rate, "ci_low": lo, "ci_high": hi,
        })

    def sort_key(r):
        try:
            return (r["policy"], float(r["trained_on"]))
        except ValueError:
            return (r["policy"], float("inf"))
    return sorted(out, key=sort_key)

def main():
    if not os.path.exists(RESULTS) or os.path.getsize(RESULTS) == 0:
        print("No results yet. Log some rollouts with eval/log_eval.py first.")
        return
    rows = load_and_aggregate()
    print(f"\n{'policy':<10}{'trained_on':<12}{'n':>4}{'success':>9}{'rate':>8}   95% CI")
    print("-" * 60)
    for r in rows:
        ci = f"[{r['ci_low']*100:4.0f}%, {r['ci_high']*100:4.0f}%]"
        print(f"{r['policy']:<10}{r['trained_on']:<12}{r['n']:>4}{r['successes']:>9}"
              f"{r['rate']*100:>7.0f}%   {ci}")
    print()

if __name__ == "__main__":
    main()
