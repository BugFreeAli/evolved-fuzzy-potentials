"""Shared loaders and statistics for every script that produces a number, table or figure of the paper.
All numbers come from the per-seed validation files in ../data (programme B, 50k steps, 50 evaluation episodes per seed).
Seeds are the unit of analysis; two-sided exact Mann-Whitney U tests; percentile bootstrap over seeds."""
import json, os
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
from scipy.stats import mannwhitneyu

# paper label -> (files, label key inside the JSON summary)
SOURCES = {
    "B0":      (["validation_B0_her.json", "validation_B0_her_seeds4-7.json"], "B0"),
    "B1":      (["validation_B1_her.json", "validation_B1_her_seeds3-6.json"], "B1"),
    "B2a":     (["validation_B2a_her.json", "validation_B2a_her_seeds4-7.json"], "B2a"),
    "B2b":     (["validation_B2b_her.json", "validation_B2b_her_seeds4-7.json"], "B2b"),
    "Ours-S1": (["validation_OursRaw_her_search1.json", "validation_OursRaw_her_search1_seeds5-6.json"], "OursRaw"),
    "Ours-S2": (["validation_OursRaw_search_raw1.json"], "Ours_her_search_raw1"),
    "C1":      (["validation_BoundedAdd_seeds0-3.json", "validation_BoundedAdd_seeds4-7.json"], "BoundedAdd"),
    "C2":      (["validation_LinearAdd_seeds0-3.json", "validation_LinearAdd_seeds4-7.json"], "LinearAdd"),
    "C3":      (["validation_BoundedPBRS_seeds0-3.json", "validation_BoundedPBRS_seeds4-7.json"], "BoundedPBRS"),
}
ORDER = list(SOURCES)                       # B0 B1 B2a B2b Ours-S1 Ours-S2 C1 C2 C3
ORIGINAL = ORDER[:6]                        # the first six methods
CONTROLS = ORDER[6:]                        # C1 bounded+additive, C2 linear+additive, C3 bounded+PBRS
EXPECTED_N = {"B0": 8, "B1": 7, "B2a": 8, "B2b": 8, "Ours-S1": 7, "Ours-S2": 4, "C1": 8, "C2": 8, "C3": 8}
# design: (shape, form) of every method
SHAPE_FORM = {"B1": ("linear", "PBRS"), "C2": ("linear", "additive"),
              "C3": ("bounded", "PBRS"), "C1": ("bounded", "additive"),
              "B2b": ("fuzzy", "PBRS"), "B2a": ("fuzzy", "additive")}
FORM_LABEL = {"B0": "none", "B1": "PBRS, linear", "B2a": "additive, fuzzy", "B2b": "PBRS, fuzzy",
              "Ours-S1": "additive, evolved", "Ours-S2": "additive, evolved",
              "C1": "additive, bounded", "C2": "additive, linear", "C3": "PBRS, bounded"}

def load(val_dir):
    out = {}
    for k, (files, key) in SOURCES.items():
        v = []
        for f in files:
            j = json.load(open(os.path.join(val_dir, f)))
            assert j["status"] == "complete", f
            assert j["training_steps"] == 50000, f
            v += [round(100 * r["success_rate"]) for r in sorted(j["summary"][key]["raw"], key=lambda r: r["seed"])]
        assert len(v) == EXPECTED_N[k], (k, len(v))
        out[k] = v
    return out

def hu(x, nd=1):
    """round half up, as everywhere in the paper"""
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))

def boot_ci(v, n=10000):                   # identical to the original analyze_results.py (one stream, seed 0)
    rng = np.random.default_rng(0)
    m = [np.mean(rng.choice(v, len(v))) for _ in range(n)]
    return np.percentile(m, [2.5, 97.5])

def p_exact(a, b):
    return float(mannwhitneyu(a, b, alternative="two-sided", method="exact").pvalue)

def hl(a, b):                              # Hodges-Lehmann shift: median of all pairwise differences
    return float(np.median(np.subtract.outer(np.asarray(a, float), np.asarray(b, float))))

def prob_sup(a, b):                        # P(A > B) + 0.5 P(A = B)
    d = np.subtract.outer(np.asarray(a, float), np.asarray(b, float))
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)

def effect(a, b, n=10000, seed=0):         # identical to the original make_comparison_figures.effect
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    md, h = [], []
    for _ in range(n):
        ra, rb = rng.choice(a, len(a)), rng.choice(b, len(b))
        md.append(ra.mean() - rb.mean()); h.append(hl(ra, rb))
    return dict(md=a.mean() - b.mean(), md_ci=np.percentile(md, [2.5, 97.5]), hl=hl(a, b),
                hl_ci=np.percentile(h, [2.5, 97.5]), ps=prob_sup(a, b), p=p_exact(a, b))

# ---- stratified contrasts (post hoc): the same contrast within each stratum, averaged over strata ----
def strat_stat(pairs):
    return float(np.mean([np.mean(a) - np.mean(b) for a, b in pairs]))

def strat_perm_p(pairs, n=200000, seed=12345):
    """Monte-Carlo permutation test, labels permuted within every stratum, two-sided on the mean of the
    per-stratum mean differences. p = (hits + 1) / (n + 1)."""
    rng = np.random.default_rng(seed)
    obs = strat_stat(pairs)
    pooled = [np.asarray(list(a) + list(b), float) for a, b in pairs]
    na = [len(a) for a, _ in pairs]
    hits = 0
    for _ in range(n):
        s = 0.0
        for x, k in zip(pooled, na):
            pm = rng.permutation(x)
            s += pm[:k].mean() - pm[k:].mean()
        if abs(s / len(pairs)) >= abs(obs) - 1e-12:
            hits += 1
    return (hits + 1) / (n + 1)

def strat_boot_ci(pairs, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    vals = [np.mean([rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for a, b in pairs]) for _ in range(n)]
    return np.percentile(vals, [2.5, 97.5])

def strat_contrasts(D):
    """name -> list of (A, B) per stratum.  Strata are the levels of the factor that is held fixed."""
    return {
        "form (additive $-$ PBRS), 3 shapes": [(D["C2"], D["B1"]), (D["C1"], D["C3"]), (D["B2a"], D["B2b"])],
        "bounded $-$ linear, 2 forms":       [(D["C1"], D["C2"]), (D["C3"], D["B1"])],
        "fuzzy $-$ bounded, 2 forms":        [(D["B2a"], D["C1"]), (D["B2b"], D["C3"])],
        "fuzzy $-$ linear, 2 forms":         [(D["B2a"], D["C2"]), (D["B2b"], D["B1"])],
        "bounded or fuzzy $-$ linear, 2 forms": [(D["C1"] + D["B2a"], D["C2"]), (D["C3"] + D["B2b"], D["B1"])],
    }
