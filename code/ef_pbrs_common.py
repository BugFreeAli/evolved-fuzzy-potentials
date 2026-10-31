"""
EF-PBRS Phase 3 -- shared infrastructure for shaped baselines (B1, B2a, B2b).

"""

import numpy as np
import gymnasium as gym





GAMMA = 0.99


def extract_features(obs: dict) -> tuple[float, float]:
    """x1 = gripper-to-object distance, x2 = object-to-goal distance.

    Confirmed empirically in Phase 3 (not just from source): observation[0:3] is
    always the end-effector position; achieved_goal tracks the object itself for
    Push/Slide/PickAndPlace/Stack, and tracks the gripper itself for Reach (there
    being no object), which collapses x1 to exactly 0 for Reach with no special
    case in this code.
    """
    ee_pos = np.asarray(obs["observation"])[0:3]
    achieved = np.asarray(obs["achieved_goal"])
    desired = np.asarray(obs["desired_goal"])
    x1 = float(np.linalg.norm(ee_pos - achieved))
    x2 = float(np.linalg.norm(achieved - desired))
    return x1, x2


def classical_potential(obs: dict) -> float:
    """B1: hand-designed potential, Phi = -x1 - x2 (Phase 2 doc, Section 5.1)."""
    x1, x2 = extract_features(obs)
    return -x1 - x2


class PotentialBasedShapingWrapper(gym.Wrapper):
    """
    Generic potential-based reward shaping: r' = r + gamma*Phi(s') - Phi(s).

    Phi(s') is forced to exactly 0 ONLY at genuine termination (terminated=True,
    e.g. task success) -- NOT at time-limit truncation (truncated=True). A time
    limit is a simulation artifact, not a real end of the task: the process
    conceptually continues past it, just unsimulated, which is exactly why SAC's
    own critic bootstraps its value estimate *through* truncation rather than
    treating it as zero-value. Zeroing Phi at truncation too (an earlier version
    of this wrapper did this) manufactures an artificial reward bonus on every
    failed/timed-out episode, sized by how far from the goal the agent still is
    -- confirmed directly from a Phase 3 diagnostic run, where every timeout
    showed a reward *bonus* instead of the honest -1. Evaluating Phi normally at
    truncation keeps our shaping consistent with how the critic already treats
    it, and removes the artifact.
    """

    def __init__(self, env, potential_fn, gamma=GAMMA):
        super().__init__(env)
        self.potential_fn = potential_fn
        self.gamma = gamma
        self._last_obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        phi_s = self.potential_fn(self._last_obs)
        phi_s_next = 0.0 if terminated else self.potential_fn(obs)

        shaped_reward = reward + self.gamma * phi_s_next - phi_s
        self._last_obs = obs
        return obs, shaped_reward, terminated, truncated, info


class RawAdditiveShapingWrapper(gym.Wrapper):
    """
    B2a: raw additive shaping, r' = r + shaping_fn(s') -- the FARS/Bingol-style
    approach, confirmed directly from Phase 1's literature review. NOT
    potential-based: no discount factor, no Phi(s) subtraction, just the raw
    shaping value at the NEW state added directly onto the reward. Included
    specifically to replicate "how the field currently does it" -- this
    wrapper deliberately carries NO policy-invariance guarantee (Proposition 1
    requires the potential-difference form in PotentialBasedShapingWrapper
    above; this is the contrast case).
    """

    def __init__(self, env, shaping_fn):
        super().__init__(env)
        self.shaping_fn = shaping_fn

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaped_reward = reward + self.shaping_fn(obs)
        return obs, shaped_reward, terminated, truncated, info










FUZZY_MF_PARAMS = {
    "Near":   {"c": 0.02, "sigma": 0.05},
    "Medium": {"c": 0.15, "sigma": 0.08},
    "Far":    {"c": 0.35, "sigma": 0.10},
}





FUZZY_RULE_TABLE = {
    ("Near",   "Near"):   0.00,
    ("Near",   "Medium"): -0.15,
    ("Near",   "Far"):    -0.35,
    ("Medium", "Near"):   -0.15,
    ("Medium", "Medium"): -0.30,
    ("Medium", "Far"):    -0.50,
    ("Far",    "Near"):   -0.35,
    ("Far",    "Medium"): -0.50,
    ("Far",    "Far"):    -0.70,
}


def _gaussian_membership(x: float, c: float, sigma: float) -> float:
    return float(np.exp(-((x - c) ** 2) / (2 * sigma ** 2)))


def _memberships(x: float) -> dict:
    return {term: _gaussian_membership(x, p["c"], p["sigma"]) for term, p in FUZZY_MF_PARAMS.items()}


def fuzzy_potential_from_features(x1: float, x2: float) -> float:
    """Core TSK computation given x1/x2 directly -- factored out from
    hand_crafted_fuzzy_potential so it's testable independently of a real
    observation dict (see diagnose_fuzzy.py)."""
    m1 = _memberships(x1)
    m2 = _memberships(x2)

    numerator = 0.0
    denominator = 0.0
    for term1, mu1 in m1.items():
        for term2, mu2 in m2.items():
            weight = mu1 * mu2
            numerator += weight * FUZZY_RULE_TABLE[(term1, term2)]
            denominator += weight

    return numerator / (denominator + 1e-8)


def hand_crafted_fuzzy_potential(obs: dict) -> float:
    """B2a/B2b's potential function. Same signature as classical_potential so
    both plug into either wrapper class interchangeably."""
    x1, x2 = extract_features(obs)
    return fuzzy_potential_from_features(x1, x2)













_TERMS = ["Near", "Medium", "Far"]





INITIAL_THETA = [
    0.02, 0.05,  0.15, 0.08,  0.35, 0.10,   
    0.02, 0.05,  0.15, 0.08,  0.35, 0.10,   
    0.00, -0.15, -0.35,                     
    -0.15, -0.30, -0.50,                    
    -0.35, -0.50, -0.70,                    
]




THETA_BOUNDS = (
    [0.0, 0.01] * 6 + [-1.5] * 9,   
    [0.6, 0.30] * 6 + [0.2] * 9,    
)


def unpack_theta(theta):
    """Splits a 21-dim theta vector into per-input MF params and a rule table."""
    mf_x1 = {_TERMS[i]: {"c": theta[2 * i], "sigma": theta[2 * i + 1]} for i in range(3)}
    mf_x2 = {_TERMS[i]: {"c": theta[6 + 2 * i], "sigma": theta[6 + 2 * i + 1]} for i in range(3)}
    rule_table = {}
    idx = 12
    for t1 in _TERMS:
        for t2 in _TERMS:
            rule_table[(t1, t2)] = theta[idx]
            idx += 1
    return mf_x1, mf_x2, rule_table


def evolved_fuzzy_potential_from_features(x1: float, x2: float, theta) -> float:
    """Same TSK computation as fuzzy_potential_from_features, but with
    per-input MF params and rule table unpacked from theta instead of the
    fixed hand-crafted globals."""
    mf_x1, mf_x2, rule_table = unpack_theta(theta)
    m1 = {t: _gaussian_membership(x1, p["c"], p["sigma"]) for t, p in mf_x1.items()}
    m2 = {t: _gaussian_membership(x2, p["c"], p["sigma"]) for t, p in mf_x2.items()}

    numerator = 0.0
    denominator = 0.0
    for term1, mu1 in m1.items():
        for term2, mu2 in m2.items():
            weight = mu1 * mu2
            numerator += weight * rule_table[(term1, term2)]
            denominator += weight

    return numerator / (denominator + 1e-8)


def make_evolved_potential_fn(theta):
    """Returns a potential_fn(obs)->float closure with theta baked in, so a
    candidate solution plugs directly into PotentialBasedShapingWrapper the
    same way classical_potential and hand_crafted_fuzzy_potential do."""
    def potential_fn(obs):
        x1, x2 = extract_features(obs)
        return evolved_fuzzy_potential_from_features(x1, x2, theta)
    return potential_fn


def evaluate_theta_fitness(theta, env_id, training_budget, n_eval_episodes, seed, distance_weight=1.0):
    """The inner loop (Phase 2 doc Section 4.2): train a fresh SAC policy
    under theta's evolved fuzzy potential for `training_budget` steps, then
    evaluate on the RAW env (no shaping -- prevents CMA-ES from finding a
    theta that games its own shaping term instead of helping the real task).

    Returns blended fitness = success_rate - distance_weight * mean_final_x2,
    not success rate alone -- confirmed necessary from the baseline data:
    even 100k steps barely clears 5-10% success on Push, so a short proxy
    budget will show ~0% for nearly every candidate, and the continuous
    distance term is what actually gives CMA-ES a gradient to search over.
    mean_final_x2 uses x2 specifically (object-to-goal distance) since that's
    the exact quantity is_success is computed from.
    """
    import gymnasium as gym
    import panda_gym  
    from stable_baselines3 import SAC
    from stable_baselines3.common.monitor import Monitor

    potential_fn = make_evolved_potential_fn(theta)
    env = Monitor(PotentialBasedShapingWrapper(gym.make(env_id), potential_fn=potential_fn, gamma=GAMMA))
    model = SAC(policy="MultiInputPolicy", env=env, seed=seed, gamma=GAMMA, verbose=0)
    model.learn(total_timesteps=training_budget)

    eval_env = gym.make(env_id)
    successes = 0
    final_distances = []
    for ep in range(n_eval_episodes):
        obs, info = eval_env.reset(seed=2000 + ep)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
        _, x2_final = extract_features(obs)
        final_distances.append(x2_final)
        if info.get("is_success", False):
            successes += 1

    success_rate = successes / n_eval_episodes
    mean_final_distance = sum(final_distances) / len(final_distances)
    fitness = success_rate - distance_weight * mean_final_distance
    return fitness, success_rate, mean_final_distance


def evaluate_theta_fitness_2seed(theta, env_id, training_budget, n_eval_episodes, seed_pair, distance_weight=1.0):
    """Averages fitness over 2 independent training seeds per candidate.

    Isolates exactly one variable against the original evolve_real_run.py
    results: same population (6) and generations (10), only the per-candidate
    seed count changes from 1 to 2. Directly tests whether per-candidate
    noise explains the first search's stagnate-then-spike pattern and the
    winner's-curse gap between search-time fitness (-0.024) and honest
    validation (7% pooled, indistinguishable from B2b's 6%) -- rather than
    conflating that test with a population or generation-count change too.
    """
    fitnesses, successes, distances = [], [], []
    for seed in seed_pair:
        fitness, succ, dist = evaluate_theta_fitness(theta, env_id, training_budget, n_eval_episodes, seed, distance_weight)
        fitnesses.append(fitness)
        successes.append(succ)
        distances.append(dist)
        print(f"      seed {seed}: fitness={fitness:.3f} (success={succ:.0%}, dist={dist:.3f})")
    avg_fitness = sum(fitnesses) / len(fitnesses)
    avg_success = sum(successes) / len(successes)
    avg_distance = sum(distances) / len(distances)
    return avg_fitness, avg_success, avg_distance
