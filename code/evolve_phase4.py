"""
EF-PBRS Phase 4 -- evolutionary search, parallelized and parameterized.

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
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import cma

from ef_pbrs_common import (
    evaluate_theta_fitness,
    evaluate_theta_fitness_2seed,
    INITIAL_THETA,
    THETA_BOUNDS,
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


def _worker_init():
    
    import torch
    torch.set_num_threads(1)
    import numpy as np
    np.seterr(all="warn")


def _evaluate_job(job: EvalJob) -> dict:
    
    fitness, success, dist = evaluate_theta_fitness(
        job.theta, job.env_id, job.training_budget, job.n_eval_episodes, job.training_seed
    )
    return {
        "generation": job.generation,
        "candidate_idx": job.candidate_idx,
        "seed_idx": job.seed_idx,
        "training_seed": job.training_seed,
        "fitness": fitness,
        "success": success,
        "distance": dist,
    }


def make_pool(max_workers: int) -> ProcessPoolExecutor:
    
    ctx = mp.get_context("spawn")  
    return ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx, initializer=_worker_init)


def evaluate_jobs_parallel(jobs: list[EvalJob], pool: ProcessPoolExecutor) -> list[dict]:
    
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
                f"seed_idx={job.seed_idx} seed={job.training_seed} theta={job.theta}"
            ) from e
    return [results_by_position[i] for i in range(len(jobs))]


def average_candidate_results(seed_results: list[dict]) -> dict:
    
    avg_success = sum(r["success"] for r in seed_results) / len(seed_results)
    avg_distance = sum(r["distance"] for r in seed_results) / len(seed_results)
    avg_fitness = sum(r["fitness"] for r in seed_results) / len(seed_results)
    return {"fitness": avg_fitness, "success": avg_success, "distance": avg_distance}


def _isclose(a: float, b: float, tol: float = 1e-6) -> bool:
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


def validate_parallel_matches_serial(env_id: str, n_eval_episodes: int, small_budget: int = 2000):
    
    theta = list(INITIAL_THETA)
    seed_pair = [999999, 1000000]  

    print(f"Validating on this machine (budget={small_budget} steps, seed_pair={seed_pair})...")

    print("  Running canonical serial reference (evaluate_theta_fitness_2seed)...")
    t0 = time.time()
    ref_fitness, ref_success, ref_dist = evaluate_theta_fitness_2seed(
        theta, env_id, small_budget, n_eval_episodes, seed_pair
    )
    print(f"    done in {time.time() - t0:.0f}s: fitness={ref_fitness:.4f}")

    print("  Running through the parallel pool (max_workers=2)...")
    jobs = [
        EvalJob(generation=-1, candidate_idx=0, seed_idx=idx, theta=theta,
                training_seed=seed, training_budget=small_budget,
                env_id=env_id, n_eval_episodes=n_eval_episodes)
        for idx, seed in enumerate(seed_pair)
    ]
    t0 = time.time()
    with make_pool(max_workers=2) as pool:
        pooled_results = evaluate_jobs_parallel(jobs, pool)
    pooled = average_candidate_results(pooled_results)
    print(f"    done in {time.time() - t0:.0f}s: fitness={pooled['fitness']:.4f}")

    checks = {
        "fitness": (ref_fitness, pooled["fitness"]),
        "success": (ref_success, pooled["success"]),
        "distance": (ref_dist, pooled["distance"]),
    }
    mismatches = {k: (a, b) for k, (a, b) in checks.items() if not _isclose(a, b)}
    if mismatches:
        raise AssertionError(
            f"Serial/parallel mismatch beyond tolerance, do not proceed: {mismatches}\n"
            "This likely means a real bug (seed not reaching the worker, a dropped "
            "job, a config value not passed through) rather than floating-point "
            "noise -- SAC training is sensitive enough to its seed that a genuine "
            "plumbing bug should produce a difference much larger than 1e-6."
        )
    print("PASSED: serial and parallel paths agree within tolerance. Safe to scale up.\n")


def make_random_theta(rng: random.Random) -> list:
    
    lower, upper = THETA_BOUNDS
    theta = []
    for lo, hi in zip(lower, upper):
        midpoint = (lo + hi) / 2
        value = rng.gauss(midpoint, 0.05)
        theta.append(min(max(value, lo), hi))
    return theta


def run_evolution(args):
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

    
    
    
    
    
    
    with make_pool(max_workers=args.max_workers) as pool:
        for gen in range(args.generations):
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

            with open(f"{args.checkpoint_dir}/evolution_log_{args.run_name}.json", "w") as f:
                json.dump(generation_log, f, indent=2)
            with open(f"{args.checkpoint_dir}/best_theta_{args.run_name}.json", "w") as f:
                json.dump({"theta": best_theta_ever, "fitness": best_fitness_ever}, f, indent=2)

            print(f"Generation {gen} done in {gen_elapsed/60:.1f} min: "
                  f"best={max(fitnesses):.3f}, mean={sum(fitnesses)/len(fitnesses):.3f}, "
                  f"best-ever={best_fitness_ever:.3f}  (checkpoint saved)")

    print(f"\n=== Run '{args.run_name}' complete ===")
    print(f"Best fitness found: {best_fitness_ever:.4f}")
    print(f"Saved to best_theta_{args.run_name}.json / evolution_log_{args.run_name}.json")
    print(f"\nNext step: validate this theta at the B0-B2b protocol (100k steps, "
          f"3+ seeds) via train_ours.py, pointing BEST_THETA_PATH at "
          f"best_theta_{args.run_name}.json.")


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--popsize", type=int, default=6,
                   help="CMA-ES population size. Default 6 reproduces the original v2 run. "
                        "Use 13 for Phase C's properly-sized-population run (matches CMA-ES's "
                        "own default heuristic lambda=4+floor(3*ln(21))=13).")
    p.add_argument("--generations", type=int, default=10)
    p.add_argument("--training-budget", type=int, default=30_000)
    p.add_argument("--n-eval-episodes", type=int, default=15)
    p.add_argument("--env-id", type=str, default="PandaPush-v3")
    p.add_argument("--seed-base", type=int, default=10_000,
                   help="Base for the per-candidate seed numbering scheme. Use a distinct "
                        "range per run (e.g. 20000 for the popsize=13 run, 30000 for the "
                        "random-init run) so different experiments' training seeds never "
                        "collide or overlap.")
    p.add_argument("--init", choices=["hand-crafted", "random"], default="hand-crafted",
                   help="'hand-crafted' matches Phase 4's original choice (seeded from "
                        "INITIAL_THETA). 'random' is the random-init ablation -- see "
                        "make_random_theta()'s docstring for the resolved interpretation "
                        "(Gaussian around bound-midpoint, std=0.05).")
    p.add_argument("--max-workers", type=int, default=(os.cpu_count() or 4),
                   help="Defaults to all detected cores. Per-seed job granularity means "
                        "12 jobs/generation at popsize=6, which divides evenly across 4 "
                        "workers -- if you want to leave a core free for the OS during a "
                        "long unattended run, pass 3 instead.")
    p.add_argument("--run-name", type=str, default="v2_parallel",
                   help="Distinguishes checkpoint filenames across configurations, e.g. "
                        "'pop13' or 'random_init', so different Phase C runs never "
                        "overwrite each other's checkpoints.")
    p.add_argument("--checkpoint-dir", type=str, default=".")
    p.add_argument("--validate-only", action="store_true",
                   help="Run only the serial-vs-parallel validation check and exit.")
    p.add_argument("--skip-validate", action="store_true",
                   help="Skip the (now genuinely cheap, ~2-4 min) validation check and "
                        "go straight to the real run. Only use this once you've already "
                        "validated successfully on this exact machine.")
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    print(f"CPU cores available on this machine: {os.cpu_count()}")

    if args.validate_only:
        validate_parallel_matches_serial(args.env_id, args.n_eval_episodes)
    elif args.skip_validate:
        run_evolution(args)
    else:
        validate_parallel_matches_serial(args.env_id, args.n_eval_episodes)
        run_evolution(args)
