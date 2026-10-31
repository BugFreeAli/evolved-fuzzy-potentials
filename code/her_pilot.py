"""
EF-PBRS -- HER + hyperparameter pilot (Phase 0, methodology-correction check).

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






CONDITIONS = {
    "current": dict(
        use_her=False,
        gamma=0.99,
        net_arch=None,       
        n_critics=2,         
        batch_size=256,      
        learning_rate=3e-4,  
        learning_starts=100, 
    ),
    "her_only": dict(
        use_her=True,
        gamma=0.99,
        net_arch=None,
        n_critics=2,
        batch_size=256,
        learning_rate=3e-4,
        learning_starts=100,
    ),
    "her_tuned": dict(
        use_her=True,
        gamma=0.95,
        net_arch=[512, 512, 512],
        n_critics=2,
        batch_size=2048,
        learning_rate=1e-3,
        learning_starts=1000,
    ),
}


@dataclass
class PilotJob:
    condition: str
    seed: int
    env_id: str
    training_steps: int
    n_eval_episodes: int
    eval_seed_offset: int


def _worker_init():
    """Same belt-and-suspenders pattern as the rest of this project's
    parallel scripts: confirms thread pinning took inside this worker."""
    import torch
    torch.set_num_threads(1)
    import numpy as np
    np.seterr(all="warn")


def _run_one(job: PilotJob) -> dict:
    import gymnasium as gym
    import panda_gym  
    from stable_baselines3 import SAC, HerReplayBuffer
    from stable_baselines3.common.monitor import Monitor

    cfg = CONDITIONS[job.condition]

    env = Monitor(gym.make(job.env_id))

    policy_kwargs = {"n_critics": cfg["n_critics"]}
    if cfg["net_arch"] is not None:
        policy_kwargs["net_arch"] = cfg["net_arch"]

    sac_kwargs = dict(
        policy="MultiInputPolicy",
        env=env,
        seed=job.seed,
        gamma=cfg["gamma"],
        batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        learning_starts=cfg["learning_starts"],
        policy_kwargs=policy_kwargs,
        verbose=0,
    )

    if cfg["use_her"]:
        
        
        
        
        assert env.spec is not None and env.spec.max_episode_steps is not None, (
            "HER needs a known max episode length; PandaPush-v3 should set "
            "env.spec.max_episode_steps automatically. If this fires, pass "
            "max_episode_length explicitly in replay_buffer_kwargs."
        )
        sac_kwargs["replay_buffer_class"] = HerReplayBuffer
        sac_kwargs["replay_buffer_kwargs"] = dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
        )

    model = SAC(**sac_kwargs)
    model.learn(total_timesteps=job.training_steps)

    eval_env = gym.make(job.env_id)
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
        "condition": job.condition,
        "seed": job.seed,
        "successes": successes,
        "n_eval_episodes": job.n_eval_episodes,
        "success_rate": successes / job.n_eval_episodes,
    }


def make_pool(max_workers: int) -> ProcessPoolExecutor:
    ctx = mp.get_context("spawn")
    return ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx, initializer=_worker_init)


def run_pilot(args):
    jobs = [
        PilotJob(condition=cond, seed=seed, env_id=args.env_id,
                  training_steps=args.training_steps,
                  n_eval_episodes=args.n_eval_episodes,
                  eval_seed_offset=args.eval_seed_offset)
        for cond in CONDITIONS
        for seed in range(args.n_seeds)
    ]

    print(f"Dispatching {len(jobs)} jobs ({len(CONDITIONS)} conditions x {args.n_seeds} seeds, "
          f"{args.training_steps} steps each) across {args.max_workers} workers...\n")

    results_by_condition = {c: [] for c in CONDITIONS}
    t0 = time.time()
    with make_pool(args.max_workers) as pool:
        futures = {pool.submit(_run_one, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                r = future.result()
            except Exception as e:
                raise RuntimeError(f"Job failed: condition={job.condition} seed={job.seed}") from e
            results_by_condition[r["condition"]].append(r)
            done = sum(len(v) for v in results_by_condition.values())
            print(f"  [{r['condition']}] seed {r['seed']}: success={r['success_rate']:.0%} "
                  f"({r['successes']}/{r['n_eval_episodes']})  ({done}/{len(jobs)} done)")
            with open(args.output, "w") as f:
                json.dump({"status": "in_progress", "raw_results": results_by_condition}, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nAll jobs complete in {elapsed/60:.1f} min.\n")
    print("=" * 60)
    print(f"RESULTS at {args.training_steps} steps")
    print("=" * 60)
    summary = {}
    for cond in CONDITIONS:
        rs = sorted(results_by_condition[cond], key=lambda r: r["seed"])
        rates = [r["success_rate"] for r in rs]
        mean_rate = sum(rates) / len(rates)
        print(f"\n{cond}: per-seed {[f'{r:.0%}' for r in rates]}, mean {mean_rate:.1%}")
        summary[cond] = {"per_seed_success_rate": rates, "mean_success_rate": mean_rate, "raw": rs}

    print(f"\nReference point -- this project's existing B0 (no HER, current recipe): "
          f"8% (12/150) at 100k steps, 9.2% (23/250) at 300k steps.")
    print("If 'current' above is wildly outside that ballpark, check this script before "
          "trusting 'her_only'/'her_tuned'. If 'her_tuned' is dramatically higher than "
          "'current', HER + hyperparameters were masking your true signal all along.")

    with open(args.output, "w") as f:
        json.dump({"status": "complete", "training_steps": args.training_steps, "summary": summary}, f, indent=2)
    print(f"\nSaved to {args.output}")


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env-id", type=str, default="PandaPush-v3")
    p.add_argument("--training-steps", type=int, default=50_000)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--n-eval-episodes", type=int, default=50)
    p.add_argument("--eval-seed-offset", type=int, default=1000)
    p.add_argument("--max-workers", type=int, default=(os.cpu_count() or 4))
    p.add_argument("--output", type=str, default="her_pilot_results.json")
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    print(f"CPU cores available: {os.cpu_count()}")
    run_pilot(args)
