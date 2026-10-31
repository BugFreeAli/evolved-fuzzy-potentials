"""
Analysis of the non-fuzzy controls: BoundedAdd, LinearAdd and (if present) BoundedPBRS.

STATUS OF EACH COMPARISON (state this in the paper):
  * BoundedAdd  : pre-specified before the run (primary = vs B2a; secondary = vs B0, B1, B2b).
  * LinearAdd   : the run was planned in advance, but the comparisons below were chosen AFTER its
                  results were seen -> report them as post hoc / exploratory.
  * BoundedPBRS : pre-specified before the run (primary = vs BoundedAdd, i.e. the form effect with the
                  shape held fixed; secondary = vs B2b, B1, B0). Seeds 0-7, no extension.
Test: exact two-sided Mann-Whitney U on per-seed success rates (scipy, method="exact"), uncorrected.
"Not significant" is never read as "equivalent" (no equivalence test).

Usage (run in the folder holding all validation_*.json files):
    python analyze_controls.py
"""
import glob
import itertools
import json
import sys

import numpy as np
from scipy.stats import mannwhitneyu

SOURCES = {
    "B0":         ["validation_B0_her.json", "validation_B0_her_seeds4-7.json"],
    "B1":         ["validation_B1_her.json", "validation_B1_her_seeds3-6.json"],
    "B2a":        ["validation_B2a_her.json", "validation_B2a_her_seeds4-7.json"],
    "B2b":        ["validation_B2b_her.json", "validation_B2b_her_seeds4-7.json"],
    "BoundedAdd": sorted(glob.glob("validation_BoundedAdd*.json")),
    "LinearAdd":  sorted(glob.glob("validation_LinearAdd*.json")),
    "BoundedPBRS": sorted(glob.glob("validation_BoundedPBRS*.json")),   # optional, may be absent
}
EXPECTED_N = {"B0": 8, "B1": 7, "B2a": 8, "B2b": 8, "BoundedAdd": 8, "LinearAdd": 8, "BoundedPBRS": 8}
N_EVOLVED_NOT_LOADED = 2   # Ours-S1 and Ours-S2 are part of the paper's family of methods but not loaded here


def load(files):
    by_seed = {}
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        if d.get("status") != "complete":
            sys.exit(f"{f} is not complete (status={d.get('status')}).")
        for v in d["summary"].values():
            for r in v["raw"]:
                if r["seed"] in by_seed:
                    sys.exit(f"Duplicate seed {r['seed']} in {f}.")
                by_seed[r["seed"]] = 100.0 * r["successes"] / r["n_eval_episodes"]
    return [by_seed[s] for s in sorted(by_seed)], sorted(by_seed)


def boot_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    return np.percentile(rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1), [2.5, 97.5])


def p_exact(a, b):
    return mannwhitneyu(a, b, alternative="two-sided", method="exact").pvalue


def half_up(x):
    return np.floor(x * 10 + 0.5 + 1e-9) / 10   # the paper rounds half up to one decimal place


data = {}
for name, files in SOURCES.items():
    if not files:
        if name == "BoundedPBRS":
            print("(BoundedPBRS files not found -- skipping that block)\n")
            continue
        sys.exit(f"No result files found for {name}.")
    vals, seeds = load(files)
    if len(vals) != EXPECTED_N[name]:
        print(f"WARNING: {name} has n={len(vals)} (expected {EXPECTED_N[name]}).")
    data[name] = vals
    lo, hi = boot_ci(vals)
    print(f"{name:12s} n={len(vals)} seeds={seeds}\n             per-seed %: {[round(v, 1) for v in vals]}\n"
          f"             mean {half_up(np.mean(vals)):.1f}  SD {np.std(vals, ddof=1):.1f}  "
          f"median {np.median(vals):.1f}  95% boot CI [{lo:.1f}, {hi:.1f}]")


def block(title, tests):
    print(f"\n{title}")
    for a, b, tag in tests:
        if a in data and b in data:
            print(f"  {a:11s} vs {b:11s} [{tag:9s}] p = {p_exact(data[a], data[b]):.4f}   "
                  f"(mean diff {np.mean(data[a]) - np.mean(data[b]):+.1f} points)")


block("BoundedAdd (pre-specified):",
      [("BoundedAdd", "B2a", "PRIMARY"), ("BoundedAdd", "B0", "secondary"),
       ("BoundedAdd", "B1", "secondary"), ("BoundedAdd", "B2b", "secondary")])
block("LinearAdd (comparisons chosen after seeing results -> post hoc):",
      [("LinearAdd", "B1", "form"), ("LinearAdd", "BoundedAdd", "shape"), ("LinearAdd", "B0", "post hoc"),
       ("LinearAdd", "B2a", "post hoc"), ("LinearAdd", "B2b", "post hoc")])
block("BoundedPBRS (pre-specified):",
      [("BoundedPBRS", "BoundedAdd", "PRIMARY"), ("BoundedPBRS", "B2b", "secondary"),
       ("BoundedPBRS", "B1", "secondary"), ("BoundedPBRS", "B0", "secondary")])

print("\nForm effect (additive minus PBRS), same potential shape:")
for shape, add, pbrs in (("linear", "LinearAdd", "B1"), ("fuzzy", "B2a", "B2b"), ("bounded", "BoundedAdd", "BoundedPBRS")):
    if add in data and pbrs in data:
        print(f"  {shape:8s} {add} - {pbrs}: {np.mean(data[add]) - np.mean(data[pbrs]):+.1f} points, "
              f"p = {p_exact(data[add], data[pbrs]):.4f}")

n_methods = len(data) + N_EVOLVED_NOT_LOADED
n_pairs = n_methods * (n_methods - 1) // 2
thr = 0.05 / n_pairs
print(f"\nMultiplicity: {n_methods} methods in the paper's family (incl. Ours-S1, Ours-S2) -> {n_pairs} pairwise tests, "
      f"Bonferroni threshold 0.05/{n_pairs} = {thr:.5f}.")
p_main = p_exact(data["B2a"], data["B0"])
print(f"B2a vs B0: p = {p_main:.5f} -> {'passes' if p_main < thr else 'does NOT pass'} the all-pairs Bonferroni threshold. "
      f"(It passes the 15-test threshold 0.0033 of the original family.)")
