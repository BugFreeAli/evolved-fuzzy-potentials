"""Sensitivity check for the p-values of the paper: tie-aware exact test.

The paper's p-values come from scipy.stats.mannwhitneyu(..., method="exact"), which assumes no tied values.
Per-seed success rates are multiples of 2 points (50 episodes), so ties are common and those p-values are conservative.
Here the exact conditional permutation distribution of the rank sum (mid-ranks, i.e. tie-aware) is enumerated
by dynamic programming. A tie-corrected normal approximation is printed as well.
Usage (from the scripts folder):  python tie_aware_check.py [DATA_DIR]      (default ../../data)
"""
import itertools, os, sys
from collections import defaultdict
import numpy as np
from scipy.stats import mannwhitneyu, rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_data as pdt

D = pdt.load(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data"))

def p_tie_exact(a, b):
    x = np.array(list(a) + list(b), float); n, N = len(a), len(x)
    r2 = np.rint(rankdata(x) * 2).astype(int)           # doubled mid-ranks are integers
    dp = [defaultdict(int) for _ in range(n + 1)]; dp[0][0] = 1
    for v in r2:
        for k in range(n - 1, -1, -1):
            for s, c in list(dp[k].items()):
                dp[k + 1][s + v] += c
    tot = sum(dp[n].values()); e2 = n * (N + 1); o2 = abs(int(r2[:n].sum()) - e2)
    return sum(c for s, c in dp[n].items() if abs(s - e2) >= o2) / tot

rows = []
for a, b in itertools.combinations(D, 2):
    rows.append((a, b, pdt.p_exact(D[a], D[b]), p_tie_exact(D[a], D[b]),
                 float(mannwhitneyu(D[a], D[b], alternative="two-sided", method="asymptotic").pvalue)))
m = len(rows); thr = 0.05 / m
print(f"{m} pairs. Bonferroni threshold 0.05/{m} = {thr:.5f}\n")
print("pair          scipy-exact  tie-aware-exact  tie-corrected-normal")
for a, b, pe, pt, pa in sorted(rows, key=lambda r: r[3])[:10]:
    print(f"{a:8s}-{b:8s} {pe:10.5f} {pt:15.5f} {pa:20.5f}")
srt = sorted(r[3] for r in rows)
print("\nHolm (tie-aware exact):")
for i, p in enumerate(srt[:3]):
    print(f"  step {i+1}: p = {p:.5f}, threshold {0.05/(m-i):.5f} -> {'reject' if p <= 0.05/(m-i) else 'stop'}")
    if p > 0.05 / (m - i): break
print(f"\npairs where tie-aware p > scipy-exact p: {sum(r[3] > r[2] + 1e-12 for r in rows)} of {m}")
print("first six methods (15 pairs), Bonferroni 0.05/15 = 0.00333:")
six = [r for r in rows if r[0] in pdt.ORIGINAL and r[1] in pdt.ORIGINAL]
for a, b, pe, pt, pa in sorted(six, key=lambda r: r[3])[:3]:
    print(f"  {a}-{b}: scipy-exact {pe:.5f}, tie-aware {pt:.5f}")
