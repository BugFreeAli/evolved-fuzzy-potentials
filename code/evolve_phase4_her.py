"""
EF-PBRS Phase 4.HER -- evolutionary search under the corrected HER + tuned
hyperparameter recipe (reuses Phase 2's validated infrastructure). RESUMABLE.

"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import json
import math
import multiprocessing as mp
import pickle
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import cma

from ef_pbrs_common import INITIAL_THETA, THETA_BOUNDS, extract_features



TUNED_GAMMA = 0.95
TUNED_HPARAMS = dict(
    tau=0.05,
    learning_rate=1e-3,
    batch_size=2048,
    learning_starts=100,
    policy_kwargs=dict(net_arch=[512, 512, 512], n_critics=2),
)



RESUME_META_KEYS = (
    "env_id", "training_budget", "n_eval_episodes", "eval_seed_offset",
    "seed_base", "init", "potential_mode", "popsize",
)


@dataclass
class EvalJob:
    generation: int
    candidate_idx: int
    seed_idx: int
    theta: list
    training_seed: int
    training_budget: int
    env_id: str
    n_eval_episodes: int
    eval_seed_offset: int
    potential_mode: str  
    
    
    


def _worker_init():
    import torch
    torch.set_num_threads(1)
    import numpy as np
    np.seterr(all="warn")


def _evaluate_job(job: EvalJob) -> dict:
    """One (candidate, seed) evaluation under the HER + tuned recipe:
    train fresh, evaluate on the RAW (unshaped) env, same
    distance_weight=1.0 blended-fitness convention as the original
    evaluate_theta_fitness in ef_pbrs_common.py (kept unchanged -- still
    needed for a usable CMA-ES gradient at these success rates)."""
    import gymnasium as gym
    import panda_gym  
    from sb3_contrib.common.wrappers import TimeFeatureWrapper
    from stable_baselines3 import SAC, HerReplayBuffer
    from stable_baselines3.common.monitor import Monitor
    from ef_pbrs_her_common import HERCompatiblePotentialWrapper

    base = TimeFeatureWrapper(gym.make(job.env_id))
    shaped = HERCompatiblePotentialWrapper(base, mode=job.potential_mode, gamma=TUNED_GAMMA, theta=job.theta)
    env = Monitor(shaped)

    model = SAC(
        policy="MultiInputPolicy", env=env, seed=job.training_seed,
        gamma=TUNED_GAMMA, verbose=0,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
            copy_info_dict=True,  
        ),
        buffer_size=min(1_000_000, job.training_budget),
        **TUNED_HPARAMS,
    )
    model.learn(total_timesteps=job.training_budget)

    eval_env = TimeFeatureWrapper(gym.make(job.env_id))  
    successes = 0
    final_distances = []
    for ep in range(job.n_eval_episodes):
        obs, info = eval_env.reset(seed=job.eval_seed_offset + ep)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
        _, x2_final = extract_features(obs)
        final_distances.append(x2_final)
        if info.get("is_success", False):
            successes += 1

    success_rate = successes / job.n_eval_episodes
    mean_final_distance = sum(final_distances) / len(final_distances)
    fitness = success_rate - mean_final_distance  

    return {
        "generation": job.generation, "candidate_idx": job.candidate_idx,
        "seed_idx": job.seed_idx, "training_seed": job.training_seed,
        "fitness": fitness, "success": success_rate, "distance": mean_final_distance,
    }


def make_pool(max_workers: int) -> ProcessPoolExecutor:
    ctx = mp.get_context("spawn")
    return ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx, initializer=_worker_init)


def evaluate_jobs_parallel(jobs, pool):
    results_by_position = {}
    position_of = {pool.submit(_evaluate_job, job): i for i, job in enumerate(jobs)}
    for future in as_completed(position_of):
        i = position_of[future]
        job = jobs[i]
        try:
            results_by_position[i] = future.result()
        except Exception as e:
            raise RuntimeError(
                f"Job failed: generation={job.generation} candidate={job.candidate_idx} "
                f"seed_idx={job.seed_idx} seed={job.training_seed}"
            ) from e
    return [results_by_position[i] for i in range(len(jobs))]


def average_candidate_results(seed_results):
    avg_success = sum(r["success"] for r in seed_results) / len(seed_results)
    avg_distance = sum(r["distance"] for r in seed_results) / len(seed_results)
    avg_fitness = sum(r["fitness"] for r in seed_results) / len(seed_results)
    return {"fitness": avg_fitness, "success": avg_success, "distance": avg_distance}


def _isclose(a, b, tol=0.05):
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


def run_preflight_checks(env_id, potential_mode, small_budget=2000):
    """Two independent checks, both required before trusting a real run:
    (1) reward-math correctness; (2) parallel-dispatch correctness -- the
    SAME theta/seed pair dispatched through the real pool twice,
    independently, must agree within tolerance (0.05, not exact equality
    -- bit-exact reproducibility across OS process boundaries isn't
    guaranteed by every library in the stack, e.g. PyBullet's solver)."""
    from ef_pbrs_her_common import validate_vectorized_matches_scalar, validate_her_wrapper_consistency

    print("=== Correctness: vectorized-vs-scalar potentials ===")
    validate_vectorized_matches_scalar()

    print(f"\n=== Correctness: step() vs compute_reward(), {potential_mode} mode ===")
    validate_her_wrapper_consistency(env_id, potential_mode, list(INITIAL_THETA), TUNED_GAMMA)

    print(f"\n=== Parallel dispatch sanity check (budget={small_budget} steps, mode={potential_mode}) ===")
    theta = list(INITIAL_THETA)
    seed_pair = [999999, 1000000]  
    jobs = [
        EvalJob(generation=-1, candidate_idx=0, seed_idx=idx, theta=theta,
                training_seed=seed, training_budget=small_budget,
                env_id=env_id, n_eval_episodes=15, eval_seed_offset=2000,
                potential_mode=potential_mode)
        for idx, seed in enumerate(seed_pair)
    ]
    with make_pool(max_workers=2) as pool:
        results_a = evaluate_jobs_parallel(jobs, pool)
    with make_pool(max_workers=2) as pool:
        results_b = evaluate_jobs_parallel(jobs, pool)
    avg_a = average_candidate_results(results_a)
    avg_b = average_candidate_results(results_b)
    mismatches = {
        k: (avg_a[k], avg_b[k]) for k in ("fitness", "success", "distance")
        if not _isclose(avg_a[k], avg_b[k])
    }
    if mismatches:
        raise AssertionError(
            f"Two independent dispatches of the SAME theta/seeds disagree beyond "
            f"tolerance: {mismatches} -- do not trust the parallel path yet."
        )
    print(f"PASSED: repeated dispatch of the same theta/seeds is consistent "
          f"(fitness {avg_a['fitness']:.3f} vs {avg_b['fitness']:.3f}).")
    print("\nALL PRE-FLIGHT CHECKS PASSED. Safe to run the real search.\n")


def make_random_theta(rng):
    lower, upper = THETA_BOUNDS
    theta = []
    for lo, hi in zip(lower, upper):
        midpoint = (lo + hi) / 2
        value = rng.gauss(midpoint, 0.05)
        theta.append(min(max(value, lo), hi))
    return theta


def _checkpoint_paths(args):
    base = args.checkpoint_dir
    return {
        "log": f"{base}/evolution_log_{args.run_name}.json",
        "theta": f"{base}/best_theta_{args.run_name}.json",
        "cma_state": f"{base}/cma_state_{args.run_name}.pkl",
        "meta": f"{base}/resume_meta_{args.run_name}.json",
    }


def run_evolution(args):
    paths = _checkpoint_paths(args)
    checkpoint_complete = all(os.path.exists(p) for p in paths.values())
    resuming = checkpoint_complete and not args.fresh

    if checkpoint_complete and args.fresh:
        print(f"--fresh passed: ignoring existing checkpoint for run-name '{args.run_name}' "
              f"and starting over (it will be overwritten).")

    if resuming:
        print(f"\n=== RESUMING run '{args.run_name}' from checkpoint ===")
        with open(paths["meta"]) as f:
            meta = json.load(f)
        
        
        for key in RESUME_META_KEYS:
            setattr(args, key, meta[key])
        print(f"Locked parameters restored from original run: {meta}")

        with open(paths["cma_state"], "rb") as f:
            es = pickle.loads(f.read())
        assert es.popsize == args.popsize, (
            f"Checkpoint corrupted: cma_state popsize ({es.popsize}) != "
            f"resume_meta popsize ({args.popsize})."
        )

        with open(paths["log"]) as f:
            generation_log = json.load(f)
        start_gen = len(generation_log)

        with open(paths["theta"]) as f:
            theta_ckpt = json.load(f)
        best_fitness_ever = theta_ckpt["fitness"]
        best_theta_ever = theta_ckpt["theta"]

        print(f"Resuming at generation {start_gen}/{args.generations - 1} "
              f"(best-ever so far: {best_fitness_ever:.4f}). At most the one "
              f"in-progress generation from before the interruption is redone "
              f"-- every completed generation's progress is preserved.")
        if start_gen >= args.generations:
            print("This run's generations are already complete -- nothing to resume. "
                  "Use a new --run-name (or --fresh) to start a different search.")
            return
    else:
        if args.init == "hand-crafted":
            initial_theta = list(INITIAL_THETA)
        else:
            initial_theta = make_random_theta(random.Random(args.seed_base))
            print(f"Random init (seed={args.seed_base}): {initial_theta}")

        es = cma.CMAEvolutionStrategy(
            initial_theta, 0.05,
            {"popsize": args.popsize, "bounds": list(THETA_BOUNDS), "seed": 1}
        )
        generation_log = []
        best_fitness_ever = float("-inf")
        best_theta_ever = list(initial_theta)
        start_gen = 0

        meta = {k: getattr(args, k) for k in RESUME_META_KEYS}
        with open(paths["meta"], "w") as f:
            json.dump(meta, f, indent=2)
        print(f"New run '{args.run_name}' -- locked parameters saved to {paths['meta']} "
              f"for any future resume: {meta}")

    with make_pool(max_workers=args.max_workers) as pool:
        for gen in range(start_gen, args.generations):
            print(f"\n=== Generation {gen}/{args.generations - 1} "
                  f"(popsize={args.popsize}, workers={args.max_workers}) ===")
            solutions = es.ask()

            jobs = []
            for i, theta in enumerate(solutions):
                base_seed = args.seed_base + (gen * args.popsize + i) * 2
                for seed_idx, seed in enumerate([base_seed, base_seed + 1]):
                    jobs.append(EvalJob(
                        generation=gen, candidate_idx=i, seed_idx=seed_idx,
                        theta=list(theta), training_seed=seed,
                        training_budget=args.training_budget, env_id=args.env_id,
                        n_eval_episodes=args.n_eval_episodes,
                        eval_seed_offset=args.eval_seed_offset,
                        potential_mode=args.potential_mode,
                    ))

            gen_start = time.time()
            results = evaluate_jobs_parallel(jobs, pool)
            gen_elapsed = time.time() - gen_start

            per_candidate = {i: [] for i in range(len(solutions))}
            for r in results:
                per_candidate[r["candidate_idx"]].append(r)

            fitnesses, successes_list, distances_list = [], [], []
            for i in range(len(solutions)):
                rs = sorted(per_candidate[i], key=lambda r: r["seed_idx"])
                for r in rs:
                    print(f"      seed {r['training_seed']}: fitness={r['fitness']:.3f} "
                          f"(success={r['success']:.0%}, dist={r['distance']:.3f})")
                avg = average_candidate_results(rs)
                fitnesses.append(avg["fitness"])
                successes_list.append(avg["success"])
                distances_list.append(avg["distance"])
                print(f"    candidate {i} -> averaged: fitness={avg['fitness']:.3f} "
                      f"(success={avg['success']:.0%}, mean_final_x2={avg['distance']:.3f})")

                if avg["fitness"] > best_fitness_ever:
                    best_fitness_ever = avg["fitness"]
                    best_theta_ever = list(solutions[i])
                    print(f"    -> new best-ever fitness ({avg['fitness']:.3f})")

            es.tell(solutions, [-f for f in fitnesses])

            gen_record = {
                "generation": gen,
                "best_fitness": max(fitnesses),
                "mean_fitness": sum(fitnesses) / len(fitnesses),
                "best_success_rate": successes_list[fitnesses.index(max(fitnesses))],
                "best_theta_this_gen": list(solutions[fitnesses.index(max(fitnesses))]),
                "best_fitness_ever": best_fitness_ever,
                "best_theta_ever": best_theta_ever,
                "generation_wall_time_sec": gen_elapsed,
            }
            generation_log.append(gen_record)

            with open(paths["log"], "w") as f:
                json.dump(generation_log, f, indent=2)
            with open(paths["theta"], "w") as f:
                json.dump({"theta": best_theta_ever, "fitness": best_fitness_ever}, f, indent=2)
            
            
            
            
            with open(paths["cma_state"], "wb") as f:
                f.write(es.pickle_dumps())

            print(f"Generation {gen} done in {gen_elapsed/60:.1f} min: "
                  f"best={max(fitnesses):.3f}, mean={sum(fitnesses)/len(fitnesses):.3f}, "
                  f"best-ever={best_fitness_ever:.3f}  (checkpoint saved, resumable)")

    print(f"\n=== Run '{args.run_name}' complete ===")
    print(f"Best fitness found: {best_fitness_ever:.4f}")
    print(f"Saved to {paths['theta']} / {paths['log']}")
    print(f"\nNext step: honestly validate this theta (5 seeds, same HER+tuned recipe, "
          f"official eval-seed-offset=1000/50-episode protocol) via:\n"
          f"  python validate_her_ladder.py --n-seeds 5 --training-steps 50000 "
          f"--method {args.potential_mode}:Ours_{args.run_name}:{paths['theta'].split('/')[-1]}\n"
          f"Do NOT trust this search-time fitness alone -- same reasoning as every "
          f"prior search in this project (v1/v2/pop13/random_init).")


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--popsize", type=int, default=4,
                   help="Scaled down from the original 6/13 -- see module docstring for why. "
                        "Ignored (and overridden) on resume -- taken from resume_meta instead.")
    p.add_argument("--generations", type=int, default=5)
    p.add_argument("--training-budget", type=int, default=20_000)
    p.add_argument("--n-eval-episodes", type=int, default=15)
    p.add_argument("--eval-seed-offset", type=int, default=2000)
    p.add_argument("--env-id", type=str, default="PandaPush-v3")
    p.add_argument("--seed-base", type=int, default=60_000,
                   help="Distinct from every prior run's seed range (10000/20000/30000/40000 "
                        "already used) so training seeds never collide across configs. "
                        "Ignored (and overridden) on resume.")
    p.add_argument("--init", choices=["hand-crafted", "random"], default="hand-crafted")
    p.add_argument("--potential-mode", choices=["evolved_pbrs", "evolved_raw"], default="evolved_raw",
                   help="evolved_raw (default): search directly in the raw-additive "
                        "formulation -- B2a (raw) beat B2b (PBRS) by 7 points and "
                        "her_search1's PBRS-evolved theta matched hand-crafted once "
                        "deployed as evolved_raw, so this searches where the signal "
                        "actually is instead of transplanting after the fact.")
    p.add_argument("--max-workers", type=int, default=(os.cpu_count() or 4))
    p.add_argument("--run-name", type=str, default="her_search_raw1")
    p.add_argument("--checkpoint-dir", type=str, default=".")
    p.add_argument("--validate-only", action="store_true",
                   help="Run only the pre-flight correctness checks and exit.")
    p.add_argument("--skip-validate", action="store_true",
                   help="Skip pre-flight checks and go straight to the real/resumed run. "
                        "Only use this once you've already validated successfully on this "
                        "machine (pre-flight checks are skipped on every resume automatically).")
    p.add_argument("--fresh", action="store_true",
                   help="Ignore any existing checkpoint for this --run-name and start over "
                        "from generation 0, overwriting it. Default behavior is to AUTO-RESUME "
                        "from an existing checkpoint if one is found -- this flag opts out of that.")
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    print(f"CPU cores available: {os.cpu_count()}")

    paths = _checkpoint_paths(args)
    will_resume = all(os.path.exists(p) for p in paths.values()) and not args.fresh

    if args.validate_only:
        run_preflight_checks(args.env_id, args.potential_mode)
    elif args.skip_validate or will_resume:
        
        
        
        run_evolution(args)
    else:
        run_preflight_checks(args.env_id, args.potential_mode)
        run_evolution(args)
