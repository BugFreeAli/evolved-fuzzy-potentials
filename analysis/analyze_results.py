"""Recomputes every number in the Results section from the raw validation JSON files.
Usage:  python analyze_results.py [DIR_WITH_JSONS]      (default: ../data)
Needs numpy and scipy. Seeds are the unit of analysis; tests are exact two-sided Mann-Whitney U.

Nine methods: the first six (B0, B1, B2a, B2b, Ours-S1, Ours-S2) and three controls run afterwards
(C1 bounded additive, C2 linear additive, C3 bounded PBRS). C1-C3 used training seeds 0-7, no extension.
The plan for C1 and C3 was fixed in scripts/analyze_bounded_control.py and scripts/analyze_controls.py before their
runs; the comparisons for C2 and every stratified contrast below were chosen after the data were seen (post hoc).
"""
import itertools, os, sys
import numpy as np
from scipy.stats import fisher_exact, kruskal, mannwhitneyu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_data as pdt

D = pdt.load(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data"))
p_exact, hu = pdt.p_exact, pdt.hu

print("method   n  mean    sd  median  boot95CI (half up)  per-seed")
for k, v in D.items():
    lo, hi = pdt.boot_ci(v)
    print(f"{k:8s}{len(v):2d} {np.mean(v):6.2f} {np.std(v, ddof=1):5.1f} {np.median(v):5.1f}  [{hu(lo)}, {hu(hi)}]  {v}")

# ---------------------------------------------------------------- the first six methods (15 pairs), as before
six = {k: D[k] for k in pdt.ORIGINAL}
print("\npairwise exact two-sided Mann-Whitney U, first six methods (15 tests)")
pv6 = {(a, b): p_exact(six[a], six[b]) for a, b in itertools.combinations(six, 2)}
for (a, b), p in pv6.items():
    print(f"{a:8s} vs {b:8s} p = {p:.4f}")
print("\nKruskal-Wallis over six methods:", kruskal(*six.values()))

ours = D["Ours-S1"] + D["Ours-S2"]
fuzzy = D["B2a"] + D["B2b"] + ours
ref = D["B0"] + D["B1"]
hand = D["B2a"] + D["B2b"]
print(f"\npooled evolved n={len(ours)} mean={np.mean(ours):.2f} median={np.median(ours)}")
for k in ("B0", "B1", "B2a", "B2b", "C1", "C2", "C3"):
    print(f"  evolved vs {k}: p = {p_exact(ours, D[k]):.4f}  (mean difference {np.mean(ours) - np.mean(D[k]):+.2f})")
print(f"first-stage exploratory contrast: fuzzy-shaped n={len(fuzzy)} mean={np.mean(fuzzy):.2f} vs references n={len(ref)} mean={np.mean(ref):.2f}: p = {p_exact(fuzzy, ref):.4f}")
print(f"hand-crafted only n={len(hand)} mean={np.mean(hand):.2f}: p = {p_exact(hand, ref):.4f}")
others = D["B0"] + D["B1"] + D["B2a"] + D["B2b"]
print(f"seeds at 4%: evolved {sum(1 for x in ours if x <= 4)}/{len(ours)}, others {sum(1 for x in others if x <= 4)}/{len(others)} (min of others = {min(others)}); Fisher p = {fisher_exact([[2, len(ours) - 2], [0, len(others)]]).pvalue:.4f}")
print(f"B2a - B2b = {np.mean(D['B2a']) - np.mean(D['B2b']):.2f} points; Bonferroni threshold for 15 tests = {0.05 / 15:.4f}")

srt = sorted(pv6.items(), key=lambda kv: kv[1])
print("\nHolm step-down over the 15 pairwise tests of the first six methods (alpha = 0.05):")
for i, ((a, b), p) in enumerate(srt):
    thr = 0.05 / (15 - i)
    print(f"  {a} vs {b}: p = {p:.4f}, Holm threshold {thr:.4f} -> {'reject' if p <= thr else 'stop'}")
    if p > thr: break
ref_max = max(ref)
print(f"\nreference seeds: n={len(ref)}, max={ref_max}; B2a seeds above that max: {sorted(x for x in D['B2a'] if x > ref_max)}, others: {sorted(x for x in D['B2a'] if x <= ref_max)}")
print("pilot recipe (32,20,32) vs corrected B0:", round(p_exact([32, 20, 32], D["B0"]), 3))

# ---------------------------------------------------------------- all nine methods (36 pairs)
print("\n================ nine methods ================")
pv9 = {(a, b): p_exact(D[a], D[b]) for a, b in itertools.combinations(D, 2)}
for (a, b), p in sorted(pv9.items(), key=lambda kv: kv[1])[:12]:
    print(f"{a:8s} vs {b:8s} p = {p:.5f}")
print(f"... {len(pv9)} tests; Bonferroni threshold 0.05/{len(pv9)} = {0.05 / len(pv9):.5f}; smallest p = {min(pv9.values()):.5f}")
srt9 = sorted(pv9.items(), key=lambda kv: kv[1])
for i, ((a, b), p) in enumerate(srt9):
    thr = 0.05 / (len(srt9) - i)
    print(f"  Holm step {i + 1}: {a} vs {b}: p = {p:.5f}, threshold {thr:.5f} -> {'reject' if p <= thr else 'stop'}")
    if p > thr: break
print("Kruskal-Wallis over nine methods:", kruskal(*D.values()))
print("p-values below 0.05 among the 36:", sorted((round(p, 4), a, b) for (a, b), p in pv9.items() if p < 0.05))

print("\n--- pre-specified comparisons (written down before the runs) ---")
print(f"C1 vs B2a (primary):  p = {p_exact(D['C1'], D['B2a']):.4f}, mean difference {np.mean(D['C1']) - np.mean(D['B2a']):+.2f}")
for k in ("B0", "B1", "B2b"):
    print(f"C1 vs {k} (secondary): p = {p_exact(D['C1'], D[k]):.4f}, mean difference {np.mean(D['C1']) - np.mean(D[k]):+.2f}")
print(f"C3 vs C1 (primary):   p = {p_exact(D['C3'], D['C1']):.4f}, mean difference {np.mean(D['C3']) - np.mean(D['C1']):+.2f}")
for k in ("B2b", "B1", "B0"):
    print(f"C3 vs {k} (secondary): p = {p_exact(D['C3'], D[k]):.4f}, mean difference {np.mean(D['C3']) - np.mean(D[k]):+.2f}")
print("\n--- C2 (comparisons chosen after the data were seen) ---")
for k in ("B1", "C1", "B0", "B2a", "B2b", "C3"):
    print(f"C2 vs {k}: p = {p_exact(D['C2'], D[k]):.4f}, mean difference {np.mean(D['C2']) - np.mean(D[k]):+.2f}")

print("\n--- design grid: means (PBRS | additive) and form effect ---")
for shape, (pb, ad) in {"linear": ("B1", "C2"), "bounded": ("C3", "C1"), "fuzzy": ("B2b", "B2a")}.items():
    print(f"{shape:8s} PBRS {pb}: {np.mean(D[pb]):.2f}  additive {ad}: {np.mean(D[ad]):.2f}  form effect {np.mean(D[ad]) - np.mean(D[pb]):+.2f}  p = {p_exact(D[ad], D[pb]):.4f}")
for form, (lin, bnd, fz) in {"PBRS": ("B1", "C3", "B2b"), "additive": ("C2", "C1", "B2a")}.items():
    print(f"{form:8s} linear {np.mean(D[lin]):.2f}  bounded {np.mean(D[bnd]):.2f}  fuzzy {np.mean(D[fz]):.2f}  bounded-linear {np.mean(D[bnd]) - np.mean(D[lin]):+.2f} (p = {p_exact(D[bnd], D[lin]):.4f})  fuzzy-linear {np.mean(D[fz]) - np.mean(D[lin]):+.2f} (p = {p_exact(D[fz], D[lin]):.4f})  fuzzy-bounded {np.mean(D[fz]) - np.mean(D[bnd]):+.2f} (p = {p_exact(D[fz], D[bnd]):.4f})")

print("\n--- stratified contrasts (POST HOC; Monte-Carlo permutation within strata, 200000 draws, seed 12345; bootstrap CI 10000, seed 0) ---")
for name, pairs in pdt.strat_contrasts(D).items():
    ci = pdt.strat_boot_ci(pairs)
    print(f"{name}: {pdt.strat_stat(pairs):+.2f} points, 95% CI [{hu(ci[0])}, {hu(ci[1])}], p = {pdt.strat_perm_p(pairs):.4f}")

print("\n--- the first-stage exploratory grouping, re-tested with the controls ---")
nonfuzzy_shaped = D["B1"] + D["C1"] + D["C2"] + D["C3"]
print(f"fuzzy-shaped (B2a, B2b, S1, S2) n={len(fuzzy)} mean={np.mean(fuzzy):.2f}  vs non-fuzzy shaped (B1, C1, C2, C3) n={len(nonfuzzy_shaped)} mean={np.mean(nonfuzzy_shaped):.2f}: p = {p_exact(fuzzy, nonfuzzy_shaped):.4f}")
print(f"hand-crafted fuzzy (B2a, B2b) n={len(hand)} mean={np.mean(hand):.2f} vs non-fuzzy shaped: p = {p_exact(hand, nonfuzzy_shaped):.4f}")
print(f"number of seeds: fuzzy-shaped additive {len(D['B2a']) + len(D['Ours-S1']) + len(D['Ours-S2'])} of {len(fuzzy)}")

print("\n--- seeds that collapsed (<= 4%) ---")
for k, v in D.items():
    print(f"{k:8s} lowest seed {min(v):2d}%, seeds at or below 4%: {sum(1 for x in v if x <= 4)}")
print(f"all nine methods: {sum(len(v) for v in D.values())} seeds, {sum(1 for v in D.values() for x in v if x <= 4)} at 4%; "
      f"lowest of the others = {min(x for v in D.values() for x in v if x > 4)}")

print("\nmean rounding check (half up):", {k: hu(np.mean(v)) for k, v in D.items()}, "references:", hu(np.mean(ref)), "fuzzy:", hu(np.mean(fuzzy)), "non-fuzzy shaped:", hu(np.mean(nonfuzzy_shaped)))
