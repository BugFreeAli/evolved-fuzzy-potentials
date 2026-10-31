"""
EF-PBRS Phase 4 -- Ours: evolved fuzzy potential, PBRS-wrapped, validation run.

"""

import json

import gymnasium as gym
import panda_gym  
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from ef_pbrs_common import PotentialBasedShapingWrapper, make_evolved_potential_fn, GAMMA

ENV_ID = "PandaPush-v3"
TRAINING_STEPS = 100_000
SEED = 0
N_EVAL_EPISODES = 50



BEST_THETA_PATH = "best_theta_v2.json"

with open(BEST_THETA_PATH) as f:
    evolved = json.load(f)
theta = evolved["theta"]
print(f"Loaded evolved theta (search fitness was {evolved['fitness']:.4f} -- "
      f"this run is what actually validates whether that number is real)\n")

MODEL_NAME = f"ours_v2_{ENV_ID.replace('-', '_')}"

potential_fn = make_evolved_potential_fn(theta)
env = Monitor(PotentialBasedShapingWrapper(gym.make(ENV_ID), potential_fn=potential_fn, gamma=GAMMA))
model = SAC(policy="MultiInputPolicy", env=env, seed=SEED, gamma=GAMMA, verbose=1)

print(f"Training Ours (evolved fuzzy, PBRS-wrapped) on {ENV_ID} for {TRAINING_STEPS} steps...\n")
model.learn(total_timesteps=TRAINING_STEPS)
model.save(MODEL_NAME)

eval_env = gym.make(ENV_ID)
successes = 0
episode_rewards = []

for ep in range(N_EVAL_EPISODES):
    obs, info = eval_env.reset(seed=1000 + ep)
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

print(f"\n--- Results after {TRAINING_STEPS} steps ---")
print(f"Success rate: {successes}/{N_EVAL_EPISODES} = {successes / N_EVAL_EPISODES:.0%}")
print(f"Mean episode reward: {sum(episode_rewards) / len(episode_rewards):.2f}")
print(f"Model saved to {MODEL_NAME}.zip")
print(f"\nCompare directly against B2b's pooled result: 6% (9/150 across 3 seeds).")
print("This single seed alone won't settle it -- treat this the same as B0-B2b's")
print("seed=0 results: a starting point, with seeds 1 and 2 to follow.")
