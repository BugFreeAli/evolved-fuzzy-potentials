
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

METHOD_TYPES = ("none", "classical_pbrs", "handcrafted_raw", "handcrafted_pbrs", "evolved_pbrs", "evolved_raw",
                "bounded_raw", "classical_raw", "bounded_pbrs")


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


def _build_train_env(env_id: str, method_type: str, theta):
    import gymnasium as gym
    import panda_gym  
    from sb3_contrib.common.wrappers import TimeFeatureWrapper
    from ef_pbrs_her_common import HERCompatiblePotentialWrapper

    base = TimeFeatureWrapper(gym.make(env_id))
    if method_type == "none":
        return base
    return HERCompatiblePotentialWrapper(base, mode=method_type, gamma=TUNED_GAMMA, theta=theta)


def _run_one(job: LadderJob) -> dict:
    import gymnasium as gym
    from sb3_contrib.common.wrappers import TimeFeatureWrapper
    from stable_baselines3 import SAC, HerReplayBuffer
    from stable_baselines3.common.monitor import Monitor

    env = Monitor(_build_train_env(job.env_id, job.method_type, job.theta))

    model = SAC(
        policy="MultiInputPolicy", env=env, seed=job.seed,
        gamma=TUNED_GAMMA, verbose=0,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
            copy_info_dict=True,  
        ),
        buffer_size=min(1_000_000, job.training_steps),
        **TUNED_HPARAMS,
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
        if method_type in ("evolved_pbrs", "evolved_raw"):
            if theta_path is None:
                raise ValueError(f"{method_type} method '{label}' requires THETA_PATH")
            with open(theta_path) as f:
                theta = json.load(f)["theta"]
        specs.append((label, method_type, theta))
        print(f"Method '{label}' ({method_type})")
    return specs


def run_preflight_validation(method_specs, env_id):
    from ef_pbrs_her_common import validate_vectorized_matches_scalar, validate_her_wrapper_consistency

    print("=== Step 1: vectorized-vs-scalar potential equivalence ===")
    validate_vectorized_matches_scalar()

    print("\n=== Step 2: step() vs compute_reward() consistency, per mode ===")
    for label, method_type, theta in method_specs:
        if method_type == "none":
            print(f"'{label}' (none/B0): no shaping wrapper involved, HER works natively -- skipped.")
            continue
        validate_her_wrapper_consistency(env_id, method_type, theta, TUNED_GAMMA)

    print("\nALL PRE-FLIGHT CHECKS PASSED. Safe to proceed to --smoke-test, then a real run.")


def run_validation(args):
    method_specs = load_method_specs(args.method)

    if args.validate_only:
        run_preflight_validation(method_specs, args.env_id)
        return

    jobs = []
    for label, method_type, theta in method_specs:
        for seed in range(args.seed_offset, args.seed_offset + args.n_seeds):
            jobs.append(LadderJob(
                label=label, method_type=method_type, theta=theta, seed=seed,
                env_id=args.env_id, training_steps=args.training_steps,
                n_eval_episodes=args.n_eval_episodes,
                eval_seed_offset=args.eval_seed_offset,
            ))

    print(f"\nDispatching {len(jobs)} jobs ({len(method_specs)} method(s) x {args.n_seeds} seeds, "
          f"{args.training_steps} steps each, TUNED hyperparameters + HER) "
          f"across {args.max_workers} workers...\n")

    results_by_label = {label: [] for label, _, _ in method_specs}
    t0 = time.time()
    with make_pool(args.max_workers) as pool:
        futures = {pool.submit(_run_one, job): job for job in jobs}
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
    print(f"RESULTS at {args.training_steps} steps -- TUNED hyperparameters + HER")
    print("=" * 60)
    for label, method_type, _ in method_specs:
        rs = sorted(results_by_label[label], key=lambda r: r["seed"])
        rates = [r["success_rate"] for r in rs]
        mean_rate = sum(rates) / len(rates)
        print(f"\n{label} ({method_type}): per-seed {[f'{r:.0%}' for r in rates]}, mean {mean_rate:.1%}")
        summary[label] = {"method_type": method_type, "per_seed_success_rate": rates,
                            "mean_success_rate": mean_rate, "raw": rs}

    print(f"\nReference points -- no-HER tuned B0: 10.7%. Phase 0 pilot's imperfect her_tuned+HER B0: 28%.")

    with open(args.output, "w") as f:
        json.dump({"status": "complete", "training_steps": args.training_steps, "summary": summary}, f, indent=2)
    print(f"\nSaved to {args.output}")


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", action="append", required=True,
                   help="TYPE:LABEL, repeatable. TYPE in: none (B0), classical_pbrs (B1), "
                        "handcrafted_raw (B2a), handcrafted_pbrs (B2b), evolved_pbrs, "
                        "evolved_raw (both need :THETA_PATH), bounded_raw and classical_raw "
                        "(non-fuzzy additive controls, no theta), bounded_pbrs "
                        "(the same bounded potential in PBRS form, no theta).")
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--seed-offset", type=int, default=0,
                   help="Start seeds at this value instead of 0 -- use to add fresh seeds "
                        "to an existing result without re-running ones you already have.")
    p.add_argument("--training-steps", type=int, default=50_000)
    p.add_argument("--n-eval-episodes", type=int, default=50)
    p.add_argument("--eval-seed-offset", type=int, default=1000)
    p.add_argument("--env-id", type=str, default="PandaPush-v3")
    p.add_argument("--max-workers", type=int, default=(os.cpu_count() or 4))
    p.add_argument("--output", type=str, default="validation_her_shaped.json")
    p.add_argument("--validate-only", action="store_true",
                   help="Run ONLY the pre-flight correctness checks and exit. Always run this first.")
    p.add_argument("--smoke-test", action="store_true",
                   help="Override training-steps to 2000 and n-seeds to 1 for a fast end-to-end check.")
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    if args.smoke_test:
        args.training_steps = 2000
        args.n_seeds = 1
        print("SMOKE TEST MODE: training_steps=2000, n_seeds=1\n")
    print(f"CPU cores available: {os.cpu_count()}")
    run_validation(args)
