"""Pre-push check: required files exist and the code contains the modes used for the controls."""
import os, sys
root = os.path.dirname(os.path.abspath(__file__))
need = ["ef_pbrs_common.py", "ef_pbrs_her_common.py", "validate_her_ladder.py", "validate_her_ladder_resumable.py",
        "evolve_phase4_her.py", "her_pilot.py"]
bad = [f for f in need if not os.path.exists(os.path.join(root, "code", f))]
if bad: sys.exit(f"missing in code/: {bad}")
for f in ("ef_pbrs_her_common.py", "validate_her_ladder.py"):
    s = open(os.path.join(root, "code", f), encoding="utf-8").read()
    for mode in ("bounded_raw", "classical_raw", "bounded_pbrs"):
        if mode not in s: sys.exit(f"code/{f} has no '{mode}': it is not the version that ran the controls")
print("OK: required code files present and control modes found.")
