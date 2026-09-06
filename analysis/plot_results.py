#!/usr/bin/env python3
"""Generate the data-efficiency curve and policy-comparison bar chart."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))
from success_stats import load_and_aggregate  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results")

def _err(rows):
    lower = [r["rate"] - r["ci_low"] for r in rows]
    upper = [r["ci_high"] - r["rate"] for r in rows]
    return [lower, upper]

def data_efficiency(rows, policy="act"):
    pts = [r for r in rows if r["policy"] == policy and r["trained_on"].isdigit()]
    pts.sort(key=lambda r: int(r["trained_on"]))
    if not pts:
        print(f"[skip] no numeric trained_on rows for policy={policy}")
        return
    x = [int(r["trained_on"]) for r in pts]
    y = [r["rate"] * 100 for r in pts]
    yerr = [[e * 100 for e in side] for side in _err(pts)]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(x, y, yerr=yerr, marker="o", capsize=4, lw=2)
    ax.set_xlabel("# demonstrations trained on")
    ax.set_ylabel("success rate (%)")
    ax.set_title(f"Data efficiency — {policy.upper()} on {len(pts)} demo budgets")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    for xi, yi, r in zip(x, y, pts):
        ax.annotate(f"n={r['n']}", (xi, yi), textcoords="offset points",
                    xytext=(0, 8), fontsize=8, ha="center")
    fig.tight_layout()
    path = os.path.join(OUT, "data_efficiency.png")
    fig.savefig(path, dpi=150)
    print(f"wrote {os.path.relpath(path)}")

def policy_comparison(rows, trained_on=None):
    if trained_on is None:
        from collections import Counter
        counts = Counter(r["trained_on"] for r in rows)
        if not counts:
            print("[skip] no rows to compare")
            return
        trained_on = counts.most_common(1)[0][0]
    pts = [r for r in rows if r["trained_on"] == trained_on]
    if len(pts) < 2:
        print(f"[skip] need >=2 policies at trained_on={trained_on} to compare")
        return
    pts.sort(key=lambda r: r["policy"])
    labels = [r["policy"] for r in pts]
    y = [r["rate"] * 100 for r in pts]
    yerr = [[e * 100 for e in side] for side in _err(pts)]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, y, yerr=yerr, capsize=5)
    ax.set_ylabel("success rate (%)")
    ax.set_title(f"Policy comparison @ {trained_on} demos")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(OUT, "policy_comparison.png")
    fig.savefig(path, dpi=150)
    print(f"wrote {os.path.relpath(path)}")

if __name__ == "__main__":
    rows = load_and_aggregate()
    data_efficiency(rows, policy="act")
    policy_comparison(rows)
