"""
EF-PBRS -- honest validation of ANY shaping method (baselines or evolved
thetas) at a configurable training budget, parallelized.

"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

from ef_pbrs_common import (
    GAMMA,
    PotentialBasedShapingWrapper,
    RawAdditiveShapingWrapper,
    classical_potential,
    hand_crafted_fuzzy_potential,
    make_evolved_potential_fn,
)

METHOD_TYPES = ("none", "classical_pbrs", "handcrafted_raw", "handcrafted_pbrs", "evolved_pbrs")


@dataclass
class ValidationJob:
    label: str
    method_type: str
    theta: list
    seed: int
    env_id: str
    training_steps: int
    n_eval_episodes: int
    eval_seed_offset: int
    save_models: bool


def _worker_init():
    import torch
    torch.set_num_threads(1)
    import numpy as np
    np.seterr(all="warn")


def _build_env(env_id: str, method_type: str, theta):
    """Dispatches to the correct, already-verified ef_pbrs_common building
    block for each method -- nothing about HOW a method shapes the reward
    is reimplemented here, only WHICH one gets picked."""
    import gymnasium as gym
    import panda_gym  

    base = gym.make(env_id)
    if method_type == "none":
        return base
    elif method_type == "classical_pbrs":
        return PotentialBasedShapingWrapper(base, potential_fn=classical_potential, gamma=GAMMA)
    elif method_type == "handcrafted_raw":
        return RawAdditiveShapingWrapper(base, shaping_fn=hand_crafted_fuzzy_potential)
    elif method_type == "handcrafted_pbrs":
        return PotentialBasedShapingWrapper(base, potential_fn=hand_crafted_fuzzy_potential, gamma=GAMMA)
    elif method_type == "evolved_pbrs":
        return PotentialBasedShapingWrapper(base, potential_fn=make_evolved_potential_fn(theta), gamma=GAMMA)
    else:
        raise ValueError(f"Unknown method_type: {method_type} (must be one of {METHOD_TYPES})")


def _validate_one(job: ValidationJob) -> dict:
    """Trains one fresh SAC policy under job.method_type's reward shaping
    (or none, for B0) for job.training_steps with job.seed, then evaluates
    deterministically on the RAW (unshaped) env at the official protocol's
    eval-seed convention -- same shape as the original theta-only version,
    generalized to dispatch on method_type."""
    from stable_baselines3 import SAC
    from stable_baselines3.common.monitor import Monitor

    env = Monitor(_build_env(job.env_id, job.method_type, job.theta))
    model = SAC(policy="MultiInputPolicy", env=env, seed=job.seed, gamma=GAMMA, verbose=0)
    model.learn(total_timesteps=job.training_steps)

    if job.save_models:
        model.save(f"{job.label}_seed{job.seed}_{job.env_id.replace('-', '_')}_{job.training_steps}steps")

    import gymnasium as gym
    eval_env = gym.make(job.env_id)
    successes = 0
    episode_rewards = []
    for ep in range(job.n_eval_episodes):
        obs, info = eval_env.reset(seed=job.eval_seed_offset + ep)
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            ep_reward += reward
            done = terminated or truncated
        episode_rewards.append(ep_reward)
        if info.get("is_success", False):
            successes += 1

    return {
        "label": job.label,
        "seed": job.seed,
        "successes": successes,
        "n_eval_episodes": job.n_eval_episodes,
        "success_rate": successes / job.n_eval_episodes,
        "mean_episode_reward": sum(episode_rewards) / len(episode_rewards),
    }


def make_pool(max_workers: int) -> ProcessPoolExecutor:
    ctx = mp.get_context("spawn")
    return ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx, initializer=_worker_init)


def load_method_specs(method_args):
    """Parses --method TYPE:LABEL or TYPE:LABEL:THETA_PATH.

    Splits on ":" with maxsplit=2 so a Windows theta path's own drive-
    letter colon (C:\\Users\\...) is never mistaken for a field separator
    -- splitting on every ":" (the original --theta parser's approach)
    would have broken on exactly this machine's paths."""
    specs = []
    for spec in method_args:
        parts = spec.split(":", 2)
        if len(parts) == 2:
            method_type, label = parts
            theta_path = None
        elif len(parts) == 3:
            method_type, label, theta_path = parts
        else:
            raise ValueError(f"--method must be TYPE:LABEL or TYPE:LABEL:THETA_PATH, got: {spec}")

        if method_type not in METHOD_TYPES:
            raise ValueError(f"Unknown method type '{method_type}', must be one of {METHOD_TYPES}")

        theta = None
        search_fitness = float("nan")
        if method_type == "evolved_pbrs":
            if theta_path is None:
                raise ValueError(f"evolved_pbrs method '{label}' requires a THETA_PATH: evolved_pbrs:{label}:path.json")
            with open(theta_path) as f:
                loaded = json.load(f)
            theta = loaded["theta"]
            search_fitness = loaded.get("fitness", float("nan"))
        elif theta_path is not None:
            raise ValueError(f"Method type '{method_type}' does not take a theta path (got {theta_path})")

        specs.append((label, method_type, theta, search_fitness))
        note = f" -- search-time fitness was {search_fitness:.4f}" if method_type == "evolved_pbrs" else ""
        print(f"Method '{label}' ({method_type}){note}")
    return specs


def run_validation(args):
    method_specs = load_method_specs(args.method)

    jobs = []
    for label, method_type, theta, _ in method_specs:
        for seed in range(args.n_seeds):
            jobs.append(ValidationJob(
                label=label, method_type=method_type, theta=theta, seed=seed,
                env_id=args.env_id, training_steps=args.training_steps,
                n_eval_episodes=args.n_eval_episodes,
                eval_seed_offset=args.eval_seed_offset,
                save_models=args.save_models,
            ))

    print(f"\nDispatching {len(jobs)} validation runs "
          f"({len(method_specs)} method(s) x {args.n_seeds} seeds, "
          f"{args.training_steps} steps each) across {args.max_workers} workers...\n")

    results_by_label = {label: [] for label, _, _, _ in method_specs}
    t0 = time.time()
    with make_pool(max_workers=args.max_workers) as pool:
        futures = {pool.submit(_validate_one, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                r = future.result()
            except Exception as e:
                raise RuntimeError(f"Validation job failed: label={job.label} seed={job.seed}") from e

            results_by_label[r["label"]].append(r)
            done_count = sum(len(v) for v in results_by_label.values())
            print(f"  [{r['label']}] seed {r['seed']}: "
                  f"success={r['success_rate']:.0%} ({r['successes']}/{r['n_eval_episodes']}), "
                  f"mean_reward={r['mean_episode_reward']:.2f}  ({done_count}/{len(jobs)} done)")

            with open(args.output, "w") as f:
                json.dump({"status": "in_progress", "raw_results": results_by_label}, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nAll validation runs complete in {elapsed/60:.1f} min ({elapsed/3600:.1f} hrs).\n")

    summary = {}
    print("=" * 70)
    print(f"RESULTS at {args.training_steps} steps -- per-seed (primary), pooled (secondary)")
    print("=" * 70)
    for label, method_type, _, search_fitness in method_specs:
        rs = sorted(results_by_label[label], key=lambda r: r["seed"])
        per_seed_rates = [r["success_rate"] for r in rs]
        pooled_successes = sum(r["successes"] for r in rs)
        pooled_n = sum(r["n_eval_episodes"] for r in rs)
        mean_of_seeds = sum(per_seed_rates) / len(per_seed_rates)

        fitness_note = f" (search-time fitness was {search_fitness:.4f})" if method_type == "evolved_pbrs" else ""
        print(f"\n{label}{fitness_note}:")
        for r in rs:
            print(f"  seed {r['seed']}: {r['success_rate']:.0%} "
                  f"({r['successes']}/{r['n_eval_episodes']}), reward={r['mean_episode_reward']:.2f}")
        print(f"  per-seed rates: {[f'{p:.0%}' for p in per_seed_rates]}")
        print(f"  mean-of-seeds success rate (PRIMARY statistic): {mean_of_seeds:.1%}")
        print(f"  pooled: {pooled_successes}/{pooled_n} = {pooled_successes/pooled_n:.1%} (descriptive only)")

        summary[label] = {
            "method_type": method_type,
            "search_fitness": search_fitness,
            "per_seed_success_rate": per_seed_rates,
            "mean_of_seeds_success_rate": mean_of_seeds,
            "pooled_successes": pooled_successes,
            "pooled_n": pooled_n,
            "pooled_success_rate": pooled_successes / pooled_n,
            "raw_per_seed": rs,
        }

    with open(args.output, "w") as f:
        json.dump({"status": "complete", "training_steps": args.training_steps, "summary": summary}, f, indent=2)
    print(f"\nFull results saved to {args.output}")


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", action="append", required=True,
                   help="TYPE:LABEL or TYPE:LABEL:THETA_PATH, repeatable. TYPE is one of: "
                        "none (B0), classical_pbrs (B1), handcrafted_raw (B2a), "
                        "handcrafted_pbrs (B2b), evolved_pbrs (requires THETA_PATH).")
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--training-steps", type=int, default=100_000,
                   help="Default matches the original B0-B2b protocol. Pass 300000 for the "
                        "budget-scaling test.")
    p.add_argument("--n-eval-episodes", type=int, default=50)
    p.add_argument("--eval-seed-offset", type=int, default=1000,
                   help="Held fixed at 1000 across every budget so training_steps is the "
                        "only variable that changes -- do not alter this for the budget test.")
    p.add_argument("--env-id", type=str, default="PandaPush-v3")
    p.add_argument("--max-workers", type=int, default=(os.cpu_count() or 4))
    p.add_argument("--save-models", action="store_true")
    p.add_argument("--output", type=str, default="validation_results.json",
                   help="Give this a budget-specific name (e.g. validation_results_300k.json) "
                        "so it never overwrites the existing 100k-step results file.")
    p.add_argument("--smoke-test", action="store_true",
                   help="Override training-steps to 2000 and n-seeds to 1 for a fast "
                        "end-to-end check of method dispatch before the real run.")
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    if args.smoke_test:
        args.training_steps = 2000
        args.n_seeds = 1
        print("SMOKE TEST MODE: training_steps=2000, n_seeds=1\n")
    print(f"CPU cores available on this machine: {os.cpu_count()}")
    run_validation(args)
