"""
EF-PBRS -- baseline ladder (B0-B2b) re-tested under the CORRECTED,
verified panda-gym-tuned hyperparameter recipe, WITHOUT HER.

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
    PotentialBasedShapingWrapper,
    RawAdditiveShapingWrapper,
    classical_potential,
    hand_crafted_fuzzy_potential,
    make_evolved_potential_fn,
)

METHOD_TYPES = ("none", "classical_pbrs", "handcrafted_raw", "handcrafted_pbrs", "evolved_pbrs")





TUNED_GAMMA = 0.95
TUNED_HPARAMS = dict(
    tau=0.05,
    learning_rate=1e-3,
    batch_size=2048,
    learning_starts=100,
    policy_kwargs=dict(net_arch=[512, 512, 512], n_critics=2),
)


@dataclass
class LadderJob:
    label: str
    method_type: str
    theta: list
    seed: int
    env_id: str
    training_steps: int
    n_eval_episodes: int
    eval_seed_offset: int


def _worker_init():
    import torch
    torch.set_num_threads(1)
    import numpy as np
    np.seterr(all="warn")


def _build_env(env_id: str, method_type: str, theta):
    import gymnasium as gym
    import panda_gym  
    from sb3_contrib.common.wrappers import TimeFeatureWrapper

    base = TimeFeatureWrapper(gym.make(env_id))
    if method_type == "none":
        return base
    elif method_type == "classical_pbrs":
        return PotentialBasedShapingWrapper(base, potential_fn=classical_potential, gamma=TUNED_GAMMA)
    elif method_type == "handcrafted_raw":
        return RawAdditiveShapingWrapper(base, shaping_fn=hand_crafted_fuzzy_potential)
    elif method_type == "handcrafted_pbrs":
        return PotentialBasedShapingWrapper(base, potential_fn=hand_crafted_fuzzy_potential, gamma=TUNED_GAMMA)
    elif method_type == "evolved_pbrs":
        return PotentialBasedShapingWrapper(base, potential_fn=make_evolved_potential_fn(theta), gamma=TUNED_GAMMA)
    else:
        raise ValueError(f"Unknown method_type: {method_type} (must be one of {METHOD_TYPES})")


def _validate_one(job: LadderJob) -> dict:
    import gymnasium as gym
    from sb3_contrib.common.wrappers import TimeFeatureWrapper
    from stable_baselines3 import SAC
    from stable_baselines3.common.monitor import Monitor

    env = Monitor(_build_env(job.env_id, job.method_type, job.theta))
    model = SAC(
        policy="MultiInputPolicy", env=env, seed=job.seed,
        gamma=TUNED_GAMMA, verbose=0, **TUNED_HPARAMS,
    )
    model.learn(total_timesteps=job.training_steps)

    
    
    
    
    
    eval_env = TimeFeatureWrapper(gym.make(job.env_id))
    successes = 0
    for ep in range(job.n_eval_episodes):
        obs, info = eval_env.reset(seed=job.eval_seed_offset + ep)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
        if info.get("is_success", False):
            successes += 1

    return {
        "label": job.label, "seed": job.seed, "successes": successes,
        "n_eval_episodes": job.n_eval_episodes,
        "success_rate": successes / job.n_eval_episodes,
    }


def make_pool(max_workers: int) -> ProcessPoolExecutor:
    ctx = mp.get_context("spawn")
    return ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx, initializer=_worker_init)


def load_method_specs(method_args):
    """TYPE:LABEL or TYPE:LABEL:THETA_PATH, maxsplit=2 (Windows-path-safe,
    same fix as the project's existing validate_thetas.py)."""
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
        if method_type == "evolved_pbrs":
            if theta_path is None:
                raise ValueError(f"evolved_pbrs method '{label}' requires THETA_PATH: evolved_pbrs:{label}:path.json")
            with open(theta_path) as f:
                theta = json.load(f)["theta"]
        specs.append((label, method_type, theta))
        print(f"Method '{label}' ({method_type})")
    return specs


def run_validation(args):
    method_specs = load_method_specs(args.method)

    jobs = []
    for label, method_type, theta in method_specs:
        for seed in range(args.n_seeds):
            jobs.append(LadderJob(
                label=label, method_type=method_type, theta=theta, seed=seed,
                env_id=args.env_id, training_steps=args.training_steps,
                n_eval_episodes=args.n_eval_episodes,
                eval_seed_offset=args.eval_seed_offset,
            ))

    print(f"\nDispatching {len(jobs)} jobs ({len(method_specs)} method(s) x {args.n_seeds} seeds, "
          f"{args.training_steps} steps each, TUNED hyperparameters, no HER) "
          f"across {args.max_workers} workers...\n")

    results_by_label = {label: [] for label, _, _ in method_specs}
    t0 = time.time()
    with make_pool(args.max_workers) as pool:
        futures = {pool.submit(_validate_one, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                r = future.result()
            except Exception as e:
                raise RuntimeError(f"Job failed: label={job.label} seed={job.seed}") from e
            results_by_label[r["label"]].append(r)
            done = sum(len(v) for v in results_by_label.values())
            print(f"  [{r['label']}] seed {r['seed']}: success={r['success_rate']:.0%} "
                  f"({r['successes']}/{r['n_eval_episodes']})  ({done}/{len(jobs)} done)")
            with open(args.output, "w") as f:
                json.dump({"status": "in_progress", "raw_results": results_by_label}, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nAll jobs complete in {elapsed/60:.1f} min ({elapsed/3600:.1f} hrs).\n")

    summary = {}
    print("=" * 60)
    print(f"RESULTS at {args.training_steps} steps -- TUNED hyperparameters, no HER")
    print("=" * 60)
    for label, method_type, _ in method_specs:
        rs = sorted(results_by_label[label], key=lambda r: r["seed"])
        rates = [r["success_rate"] for r in rs]
        mean_rate = sum(rates) / len(rates)
        print(f"\n{label} ({method_type}): per-seed {[f'{r:.0%}' for r in rates]}, mean {mean_rate:.1%}")
        summary[label] = {"method_type": method_type, "per_seed_success_rate": rates,
                            "mean_success_rate": mean_rate, "raw": rs}

    print(f"\nReference points -- OLD (untuned, no HER) results: B0 8-9.2%, B1 7.2-10%, "
          f"B2a 6-7.6%, B2b 6%. her_pilot.py's her_tuned+HER B0 (imperfect recipe): 28%.")

    with open(args.output, "w") as f:
        json.dump({"status": "complete", "training_steps": args.training_steps, "summary": summary}, f, indent=2)
    print(f"\nSaved to {args.output}")


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", action="append", required=True,
                   help="TYPE:LABEL, repeatable. TYPE in: none (B0), classical_pbrs (B1), "
                        "handcrafted_raw (B2a), handcrafted_pbrs (B2b), evolved_pbrs (needs :THETA_PATH).")
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--training-steps", type=int, default=50_000)
    p.add_argument("--n-eval-episodes", type=int, default=50)
    p.add_argument("--eval-seed-offset", type=int, default=1000)
    p.add_argument("--env-id", type=str, default="PandaPush-v3")
    p.add_argument("--max-workers", type=int, default=(os.cpu_count() or 4))
    p.add_argument("--output", type=str, default="validation_results_tuned.json")
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    print(f"CPU cores available: {os.cpu_count()}")
    run_validation(args)
