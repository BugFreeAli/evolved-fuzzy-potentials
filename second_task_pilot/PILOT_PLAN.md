# Second-task pilot: plan (commit this file to git BEFORE starting any run)

Question: on which second panda-gym task does the unshaped agent (B0) have usable headroom at 50k steps?
Candidates: PandaSlide-v3, PandaPickAndPlace-v3 (both have a single object, so x1 and x2 are defined as in Push).
Excluded: PandaReach (no object, trivial), PandaStack (achieved goal has two objects, x1 undefined), PandaFlip.

Protocol (identical to the main study): validate_her_ladder_resumable.py, method `none` (B0), HER + adapted recipe,
50k steps, 50 deterministic evaluation episodes, eval seed offset 1000. Training seeds 100 and 101 per task
(not 0-7, so no pilot seed is reused in a later study).

Decision rule (fixed now):
1. A task is usable if the B0 mean over its 2 seeds is between 5% and 50% success.
2. If both tasks are usable, pick the one with the B0 mean closer to 25% (middle of the usable range).
   Tie-break: the task whose two seeds are closer to each other (lower spread).
3. If neither is usable (both below 5% or both above 50%), do not run the full study at 50k steps.
   Report it as a limitation, or repeat the pilot for one task at 100k steps (about double the time).
4. Pilot results are not used as data in the paper.

Next step if a task is chosen: B0, B2a, C1 (bounded additive), C3 (bounded PBRS), 8 seeds each (seeds 0-7).
