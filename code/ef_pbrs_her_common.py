"""
EF-PBRS -- HER-compatible reward shaping (Phase 2).

"""

import math

import gymnasium as gym
import numpy as np

from ef_pbrs_common import (
    FUZZY_MF_PARAMS,
    FUZZY_RULE_TABLE,
    unpack_theta,
    fuzzy_potential_from_features,
    evolved_fuzzy_potential_from_features,
)

POTENTIAL_MODES = ("classical_pbrs", "handcrafted_pbrs", "handcrafted_raw", "evolved_pbrs", "evolved_raw",
                   "bounded_raw", "classical_raw", "bounded_pbrs")


RAW_MODES = ("handcrafted_raw", "evolved_raw", "bounded_raw", "classical_raw")




def _gaussian_membership_vec(x, c, sigma):
    return np.exp(-((x - c) ** 2) / (2 * sigma ** 2))


def classical_potential_vec(x1, x2):
    return -x1 - x2




BOUNDED_SCALE = 0.70
BOUNDED_WIDTH = 0.35


def bounded_potential_vec(x1, x2):
    return -BOUNDED_SCALE * np.tanh((x1 + x2) / BOUNDED_WIDTH)


def _tsk_vec(x1, x2, mf_x1, mf_x2, rule_table):
    m1 = {t: _gaussian_membership_vec(x1, p["c"], p["sigma"]) for t, p in mf_x1.items()}
    m2 = {t: _gaussian_membership_vec(x2, p["c"], p["sigma"]) for t, p in mf_x2.items()}
    numerator = 0.0
    denominator = 0.0
    for t1, mu1 in m1.items():
        for t2, mu2 in m2.items():
            w = mu1 * mu2
            numerator = numerator + w * rule_table[(t1, t2)]
            denominator = denominator + w
    return numerator / (denominator + 1e-8)


def fuzzy_potential_vec(x1, x2):
    return _tsk_vec(x1, x2, FUZZY_MF_PARAMS, FUZZY_MF_PARAMS, FUZZY_RULE_TABLE)


def evolved_fuzzy_potential_vec(x1, x2, theta):
    mf_x1, mf_x2, rule_table = unpack_theta(theta)
    return _tsk_vec(x1, x2, mf_x1, mf_x2, rule_table)


def validate_vectorized_matches_scalar(n_samples: int = 200, tol: float = 1e-9):
    """Cheap pre-flight check (run this before trusting any real job): the
    vectorized potentials above must be numerically identical to
    ef_pbrs_common.py's existing, already-verified scalar functions --
    same "verify cheaply before spending real compute" discipline as
    evolve_phase4.py's validate_parallel_matches_serial."""
    rng = np.random.default_rng(0)
    x1s = rng.uniform(0, 0.5, n_samples)
    x2s = rng.uniform(0, 0.5, n_samples)
    theta = rng.uniform(-0.5, 0.5, 21)
    for i in (1, 3, 5, 7, 9, 11):  
        theta[i] = abs(theta[i]) + 0.01

    vec_classical = classical_potential_vec(x1s, x2s)
    vec_fuzzy = fuzzy_potential_vec(x1s, x2s)
    vec_evolved = evolved_fuzzy_potential_vec(x1s, x2s, theta)
    vec_bounded = bounded_potential_vec(x1s, x2s)

    for i in range(n_samples):
        s_classical = -x1s[i] - x2s[i]
        s_fuzzy = fuzzy_potential_from_features(x1s[i], x2s[i])
        s_evolved = evolved_fuzzy_potential_from_features(x1s[i], x2s[i], theta)
        assert abs(vec_classical[i] - s_classical) < tol, f"classical mismatch at sample {i}"
        assert abs(vec_fuzzy[i] - s_fuzzy) < tol, f"fuzzy mismatch at sample {i}"
        assert abs(vec_evolved[i] - s_evolved) < tol, f"evolved mismatch at sample {i}"
        s_bounded = -0.70 * math.tanh((float(x1s[i]) + float(x2s[i])) / 0.35)  
        assert abs(vec_bounded[i] - s_bounded) < tol, f"bounded mismatch at sample {i}"
    print(f"PASSED: vectorized potentials match scalar ef_pbrs_common functions on {n_samples} random samples.")



class HERCompatiblePotentialWrapper(gym.Wrapper):

    def __init__(self, env, mode: str, gamma: float, theta=None):
        assert mode in POTENTIAL_MODES, f"mode must be one of {POTENTIAL_MODES}, got {mode}"
        if mode in ("evolved_pbrs", "evolved_raw"):
            assert theta is not None, f"{mode} requires theta"
        super().__init__(env)
        self.mode = mode
        self.gamma = gamma
        self.theta = theta
        self._last_obs = None

    def _potential(self, x1, x2):
        if self.mode in ("handcrafted_pbrs", "handcrafted_raw"):
            return fuzzy_potential_vec(x1, x2)
        elif self.mode in ("classical_pbrs", "classical_raw"):
            return classical_potential_vec(x1, x2)
        elif self.mode in ("bounded_raw", "bounded_pbrs"):
            return bounded_potential_vec(x1, x2)
        elif self.mode in ("evolved_pbrs", "evolved_raw"):
            return evolved_fuzzy_potential_vec(x1, x2, self.theta)
        raise ValueError(self.mode)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        prev_ee_pos = np.asarray(self._last_obs["observation"])[0:3]
        prev_achieved = np.asarray(self._last_obs["achieved_goal"])
        next_ee_pos = np.asarray(obs["observation"])[0:3]
        next_achieved = np.asarray(obs["achieved_goal"])
        desired = np.asarray(obs["desired_goal"])  

        x1_prev = float(np.linalg.norm(prev_ee_pos - prev_achieved))
        x2_prev = float(np.linalg.norm(prev_achieved - desired))
        x1_next = float(np.linalg.norm(next_ee_pos - next_achieved))
        x2_next = float(np.linalg.norm(next_achieved - desired))

        if self.mode in RAW_MODES:
            shaped_reward = float(reward) + float(self._potential(x1_next, x2_next))
        else:
            phi_s = float(self._potential(x1_prev, x2_prev))
            phi_s_next = 0.0 if terminated else float(self._potential(x1_next, x2_next))
            shaped_reward = float(reward) + self.gamma * phi_s_next - phi_s

        
        
        info["prev_ee_pos"] = prev_ee_pos
        info["prev_achieved_goal"] = prev_achieved
        info["next_ee_pos"] = next_ee_pos
        info["terminated"] = terminated

        self._last_obs = obs
        return obs, shaped_reward, terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Called by HerReplayBuffer for relabeled samples (batched,
        vectorized -- info is then a list/array of dicts), and by our own
        validate_her_wrapper_consistency check below (single dict)."""
        achieved_goal = np.asarray(achieved_goal)
        desired_goal = np.asarray(desired_goal)
        single = achieved_goal.ndim == 1
        if single:
            achieved_b = achieved_goal[None, :]
            desired_b = desired_goal[None, :]
            infos = [info]
        else:
            achieved_b = achieved_goal
            desired_b = desired_goal
            infos = info

        raw = np.asarray(
            self.env.unwrapped.compute_reward(achieved_b, desired_b, infos)
        ).astype(np.float64).reshape(-1)

        prev_ee_pos = np.stack([np.asarray(i["prev_ee_pos"]) for i in infos])
        prev_achieved = np.stack([np.asarray(i["prev_achieved_goal"]) for i in infos])
        next_ee_pos = np.stack([np.asarray(i["next_ee_pos"]) for i in infos])
        terminated = np.array([bool(i["terminated"]) for i in infos])

        x1_prev = np.linalg.norm(prev_ee_pos - prev_achieved, axis=-1)
        x2_prev = np.linalg.norm(prev_achieved - desired_b, axis=-1)
        x1_next = np.linalg.norm(next_ee_pos - achieved_b, axis=-1)
        x2_next = np.linalg.norm(achieved_b - desired_b, axis=-1)

        if self.mode in RAW_MODES:
            out = raw + self._potential(x1_next, x2_next)
        else:
            phi_s = self._potential(x1_prev, x2_prev)
            phi_s_next = np.where(terminated, 0.0, self._potential(x1_next, x2_next))
            out = raw + self.gamma * phi_s_next - phi_s

        return float(out[0]) if single else out


def validate_her_wrapper_consistency(env_id: str, mode: str, theta, gamma: float,
                                       n_steps: int = 200, tol: float = 1e-6):
    
    import gymnasium as gym
    import panda_gym  
    from sb3_contrib.common.wrappers import TimeFeatureWrapper

    base = TimeFeatureWrapper(gym.make(env_id))
    env = HERCompatiblePotentialWrapper(base, mode=mode, gamma=gamma, theta=theta)

    obs, info = env.reset(seed=123456)
    mismatches = []
    for t in range(n_steps):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        recomputed = env.compute_reward(next_obs["achieved_goal"], next_obs["desired_goal"], info)
        if not math.isclose(reward, recomputed, rel_tol=tol, abs_tol=tol):
            mismatches.append((t, reward, recomputed))
        obs = next_obs
        if terminated or truncated:
            obs, info = env.reset()

    if mismatches:
        raise AssertionError(
            f"step() vs compute_reward() MISMATCH on {len(mismatches)}/{n_steps} real "
            f"transitions (mode={mode}) -- DO NOT trust HER on this mode until fixed. "
            f"First mismatch: step_idx={mismatches[0][0]}, step()={mismatches[0][1]:.6f}, "
            f"compute_reward()={mismatches[0][2]:.6f}"
        )
    print(f"PASSED: step() and compute_reward() agree on all {n_steps} real transitions (mode={mode}).")
