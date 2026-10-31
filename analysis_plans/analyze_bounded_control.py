"""
Analysis of the bounded non-fuzzy additive control (BoundedAdd) -- written BEFORE the run.

PRE-SPECIFIED PLAN (do not change after seeing results):
  * Control = bounded_raw, 50k steps, HER + corrected recipe, training seeds 0-7, no extension.
  * PRIMARY comparison: BoundedAdd vs B2a (hand-crafted fuzzy, additive).
  * SECONDARY: BoundedAdd vs B0, B1, B2b.
  * Test: exact two-sided Mann-Whitney U on per-seed success rates (scipy, method="exact"), uncorrected.
  * "Not significant" is never read as "equivalent" (no equivalence test).

Usage (run in the folder holding all the validation_*.json files):
    python analyze_bounded_control.py
"""
import glob
import json
import sys

import numpy as np
from scipy.stats import mannwhitneyu

SOURCES = {
    "B0":  ["validation_B0_her.json", "validation_B0_her_seeds4-7.json"],
    "B1":  ["validation_B1_her.json", "validation_B1_her_seeds3-6.json"],
    "B2a": ["validation_B2a_her.json", "validation_B2a_her_seeds4-7.json"],
    "B2b": ["validation_B2b_her.json", "validation_B2b_her_seeds4-7.json"],
    "BoundedAdd": sorted(glob.glob("validation_BoundedAdd*.json")),
}
EXPECTED_N = {"B0": 8, "B1": 7, "B2a": 8, "B2b": 8, "BoundedAdd": 8}


def load(files):
    by_seed = {}
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        if d.get("status") != "complete":
            sys.exit(f"{f} is not complete (status={d.get('status')}).")
        for label, v in d["summary"].items():
            for r in v["raw"]:
                if r["seed"] in by_seed:
                    sys.exit(f"Duplicate seed {r['seed']} in {f}.")
                by_seed[r["seed"]] = 100.0 * r["successes"] / r["n_eval_episodes"]
    return [by_seed[s] for s in sorted(by_seed)], sorted(by_seed)


def boot_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


data = {}
for name, files in SOURCES.items():
    if not files:
        sys.exit(f"No result files found for {name}.")
    vals, seeds = load(files)
    if len(vals) != EXPECTED_N[name]:
        print(f"WARNING: {name} has n={len(vals)} (expected {EXPECTED_N[name]}).")
    data[name] = vals
    lo, hi = boot_ci(vals)
    print(f"{name:11s} n={len(vals)} seeds={seeds}\n            per-seed %: {[round(v, 1) for v in vals]}\n"
          f"            mean {np.mean(vals):.1f}  SD {np.std(vals, ddof=1):.1f}  median {np.median(vals):.1f}  "
          f"95% boot CI [{lo:.1f}, {hi:.1f}]")

print("\nExact two-sided Mann-Whitney U on per-seed rates (uncorrected):")
c = data["BoundedAdd"]
for ref, tag in (("B2a", "PRIMARY"), ("B0", "secondary"), ("B1", "secondary"), ("B2b", "secondary")):
    p = mannwhitneyu(c, data[ref], alternative="two-sided", method="exact").pvalue
    print(f"  BoundedAdd vs {ref:4s} [{tag:9s}] p = {p:.4f}   (mean diff {np.mean(c) - np.mean(data[ref]):+.1f} points)")
print("\nReminder: with 7 methods there are 21 pairwise tests -> Bonferroni threshold 0.05/21 = 0.0024.")
