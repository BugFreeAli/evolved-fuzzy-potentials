# evolved-fuzzy-potentials

Code, parameter vectors, per-seed results and analysis scripts for

> Gondal, A. A. and Jan, N. *Fuzzy, Bounded or Linear? A Controlled Study of Hand-Crafted and Evolved Reward-Shaping Potentials for Sparse-Reward Manipulation with Hindsight Experience Replay.* (under review)

Task: PandaPush-v3 (panda-gym), Soft Actor-Critic + Hindsight Experience Replay (Stable-Baselines3).

## Layout

| Folder | Content |
|---|---|
| `code/` | Training, validation and search scripts (see list below) |
| `data/` | Per-seed result files of programme B (all nine methods), the two evolution logs, the evolved parameter vectors |
| `data/pilot_and_meta/` | Pilot results, adapted-recipe B0 without HER, resume metadata of search S2 |
| `analysis/` | Scripts that recompute every statistic of the paper from `data/` |
| `analysis_plans/` | The analysis plans of the controls (C1: `analyze_bounded_control.py`, C2/C3: `analyze_controls.py`) |

## Code files (`code/`)

`ef_pbrs_common.py`, `ef_pbrs_her_common.py` (HER-safe wrapper, modes incl. `bounded_raw`, `classical_raw`, `bounded_pbrs`), `validate_her_ladder.py`, `validate_her_ladder_resumable.py`, `evolve_phase4_her.py` (searches S1/S2), `her_pilot.py`, `validate_ladder_tuned.py`, plus the programme A scripts.

## Environment

Python 3.11; see `requirements.txt` (versions as in the paper, Section 4.6). CPU only. A 50k-step run takes 14.5 to 19.7 h on a laptop-class CPU with four runs in parallel.

## Reproduce

Pre-flight checks (always first):

    cd code
    python validate_her_ladder.py --validate-only --method bounded_raw:C1 --method classical_raw:C2 --method bounded_pbrs:C3

One validation job set (example, control C1, seeds 0-3):

    python validate_her_ladder_resumable.py --n-seeds 4 --training-steps 50000 --method bounded_raw:BoundedAdd --output validation_BoundedAdd_seeds0-3.json

Search S2: `python evolve_phase4_her.py --run-name her_search_raw1 --seed-base 60000`

Statistics of the paper (from the result files):

    cd analysis
    python analyze_results.py ../data          # tables, p-values, intervals (compare analyze_results_output.txt)
    python tie_aware_check.py ../data          # tie-aware sensitivity check (Section 4.5)
    cd ../data && python ../analysis_plans/analyze_bounded_control.py   # plan written before the C1 run

## Notes

* `data/best_theta_her_search_raw1.json` was written from the last entry of `evolution_log_her_search_raw1.json` (`best_theta_ever`).
* The repository was created after the experiments. The analysis plans in `analysis_plans/` were written before the control runs, but git history cannot prove that order.
* PyBullet is not bit-reproducible across processes; repeated runs of one seed can differ slightly.

## License

MIT (see `LICENSE`).
