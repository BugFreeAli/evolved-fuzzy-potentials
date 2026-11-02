# Second task (PandaSlide-v3): study plan and analysis plan
# Commit this file to git BEFORE starting any run. Do not edit it afterwards.

Pilot (done, not used as data): B0 on PandaSlide-v3 = 34%, 34% (seeds 100, 101); PandaPickAndPlace-v3 = 8%, 10%.
By the pilot rule, Slide is the second task.

## Design
Same protocol as Push: SAC + HER, adapted recipe, 50k steps, 50 deterministic eval episodes, eval seed offset 1000.
Methods (labels): B0_Slide (none), B2a_Slide (handcrafted_raw), C1_Slide (bounded_raw), C3_Slide (bounded_pbrs).
Training seeds 0-7 for every method, n = 8 each, FIXED. No seed extension after looking at results.
The potentials are used exactly as on Push: no re-tuning of the fuzzy parameters or of the constants 0.70 and 0.35.
Evolved potentials are not run on Slide (they were not better on Push).

## Analysis (fixed now)
Unit: the training seed (per-seed success rate). Exact two-sided Mann-Whitney U (SciPy, method="exact"), uncorrected,
plus the exact tie-aware permutation p-value, both reported for every comparison.
PRIMARY: C1 vs B2a (same question as on Push: does fuzzy structure add anything over a bounded non-fuzzy potential?).
SECONDARY (family of 4 tests with B2a-C1; Bonferroni threshold 0.05/4 = 0.0125 is stated, not used to drop results):
  B2a vs B0, C1 vs B0, C1 vs C3 (additive vs PBRS form for the bounded potential).
"Not significant" is never read as "equivalent".
Per method: per-seed rates, mean, SD, median, 95% percentile bootstrap CI (10,000 resamples).
All four methods are reported, whatever the outcome. Nothing is dropped or re-run selectively.

## Run order
Phase 1: B0, B2a, C1 (seeds 0-3 on laptop 1, seeds 4-7 on laptop 2). Phase 2: C3 (same split).
