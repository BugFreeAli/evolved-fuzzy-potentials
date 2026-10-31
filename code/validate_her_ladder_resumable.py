"""
EF-PBRS -- validate_her_ladder.py, made crash/power-loss resumable.

"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import hashlib
import json
import multiprocessing as mp
import re
import shutil
import sys
import tempfile
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import validate_her_ladder as vhl
from validate_her_ladder import LadderJob

FAKE_ENV_ID = "FakeGoalPush-v0"


@dataclass
class ResumableJob(LadderJob):
    ckpt_dir: str
    ckpt_every: int
    fake_env: bool = False          
    hparam_override: dict = None    


class ConfigMismatch(RuntimeError):
    pass





def _fsync(path):
    with open(path, "rb+") as f:
        f.flush()
        os.fsync(f.fileno())


def _atomic_write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _hparams(job):
    hp = dict(vhl.TUNED_HPARAMS)
    if job.hparam_override:
        hp.update(job.hparam_override)
    return hp


def job_fingerprint(job) -> str:
    cfg = dict(label=job.label, method_type=job.method_type, theta=job.theta, seed=job.seed,
               env_id=job.env_id, training_steps=job.training_steps, n_eval=job.n_eval_episodes,
               eval_offset=job.eval_seed_offset, gamma=vhl.TUNED_GAMMA, hparams=_hparams(job),
               fake=job.fake_env)
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


def job_dir(job):
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", job.label)
    return os.path.join(job.ckpt_dir, f"{safe}_seed{job.seed}")





class CkptStore:
    SLOTS = ("slot0", "slot1")

    def __init__(self, directory, fingerprint):
        self.dir = directory
        self.fp = fingerprint
        os.makedirs(self.dir, exist_ok=True)

    
    def slot_dir(self, slot): return os.path.join(self.dir, slot)
    def result_path(self): return os.path.join(self.dir, "result.json")
    def final_model_path(self): return os.path.join(self.dir, "final_model.zip")
    def final_meta_path(self): return os.path.join(self.dir, "final.json")
    def log_path(self): return os.path.join(self.dir, "events.log")

    
    def log(self, **event):
        event["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_path(), "a") as f:
            f.write(json.dumps(event) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def events(self):
        out = []
        try:
            with open(self.log_path()) as f:
                for line in f:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        pass   
        except OSError:
            pass
        return out

    
    def check_fingerprint(self):
        metas = [_read_json(os.path.join(self.slot_dir(s), "meta.json")) for s in self.SLOTS]
        metas += [_read_json(self.final_meta_path()), _read_json(self.result_path())]
        for m in metas:
            if m and m.get("fingerprint") not in (None, self.fp):
                raise ConfigMismatch(
                    f"{self.dir} holds a checkpoint/result from a DIFFERENT configuration "
                    f"(fingerprint {m.get('fingerprint')} != {self.fp}). You changed steps, theta, label, "
                    f"hyperparameters or seeds. Delete or rename that folder (or use another --ckpt-dir).")

    
    def _valid_meta(self, slot):
        meta = _read_json(os.path.join(self.slot_dir(slot), "meta.json"))
        if not meta or meta.get("fingerprint") != self.fp:
            return None
        for fname, size in meta.get("sizes", {}).items():
            p = os.path.join(self.slot_dir(slot), fname)
            if not os.path.exists(p) or os.path.getsize(p) != size:
                return None
        return meta

    def latest(self):
        best = None
        for s in self.SLOTS:
            m = self._valid_meta(s)
            if m and (best is None or m["n_steps"] > best[1]["n_steps"]):
                best = (s, m)
        return best

    
    def save(self, model, n_steps):
        cur = self.latest()
        target = self.SLOTS[0] if cur is None else self.SLOTS[1 - self.SLOTS.index(cur[0])]
        d = self.slot_dir(target)
        os.makedirs(d, exist_ok=True)
        meta_path = os.path.join(d, "meta.json")
        if os.path.exists(meta_path):
            os.remove(meta_path)          

        tmp_model = os.path.join(d, "model.tmp.zip")
        model.save(tmp_model)
        _fsync(tmp_model)
        os.replace(tmp_model, os.path.join(d, "model.zip"))

        tmp_buf = os.path.join(d, "buffer.tmp.pkl")
        model.save_replay_buffer(tmp_buf)
        _fsync(tmp_buf)
        os.replace(tmp_buf, os.path.join(d, "buffer.pkl"))

        sizes = {f: os.path.getsize(os.path.join(d, f)) for f in ("model.zip", "buffer.pkl")}
        _atomic_write_json(meta_path, dict(n_steps=int(n_steps), fingerprint=self.fp, sizes=sizes,
                                           saved_at=time.strftime("%Y-%m-%d %H:%M:%S")))
        return target

    def verify(self, slot):
        """Cheap integrity check of a just-written slot (zip CRCs + buffer unpickles)."""
        import pickle
        d = self.slot_dir(slot)
        with zipfile.ZipFile(os.path.join(d, "model.zip")) as z:
            bad = z.testzip()
            if bad is not None:
                raise RuntimeError(f"corrupt member in model.zip: {bad}")
        with open(os.path.join(d, "buffer.pkl"), "rb") as f:
            pickle.load(f)

    def cleanup_slots(self):
        for s in self.SLOTS:
            shutil.rmtree(self.slot_dir(s), ignore_errors=True)





def _ensure_fake_registered():
    import gymnasium as gym
    import numpy as np
    from gymnasium import spaces

    if FAKE_ENV_ID in gym.registry:
        return

    class FakeGoalEnv(gym.Env):
        """Tiny goal-conditioned Dict-obs env (sparse reward, early termination on success)
        that satisfies HerReplayBuffer's contract. Used only by --selftest-resume."""
        metadata = {}

        def __init__(self):
            f32 = np.float32
            self.observation_space = spaces.Dict(dict(
                observation=spaces.Box(-5, 5, (4,), f32),
                achieved_goal=spaces.Box(-5, 5, (2,), f32),
                desired_goal=spaces.Box(-5, 5, (2,), f32)))
            self.action_space = spaces.Box(-1, 1, (2,), f32)
            self.thr = 0.3

        def compute_reward(self, achieved_goal, desired_goal, info=None):
            d = np.linalg.norm(np.asarray(achieved_goal) - np.asarray(desired_goal), axis=-1)
            return -(d > self.thr).astype(np.float32)

        def _obs(self):
            f32 = np.float32
            return dict(observation=np.concatenate([self.pos, self.vel]).astype(f32),
                        achieved_goal=self.pos.astype(f32), desired_goal=self.goal.astype(f32))

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.pos = self.np_random.uniform(-1, 1, 2)
            self.goal = self.np_random.uniform(-1, 1, 2)
            self.vel = np.zeros(2)
            return self._obs(), {}

        def step(self, action):
            self.vel = 0.5 * self.vel + 0.2 * np.clip(action, -1, 1)
            self.pos = np.clip(self.pos + self.vel, -2, 2)
            r = float(self.compute_reward(self.pos, self.goal))
            success = r == 0.0
            return self._obs(), r, success, False, {"is_success": success}

    gym.register(id=FAKE_ENV_ID, entry_point=FakeGoalEnv, max_episode_steps=50)


def _make_train_env(job):
    from sb3_contrib.common.wrappers import TimeFeatureWrapper
    from stable_baselines3.common.monitor import Monitor
    if job.fake_env:
        import gymnasium as gym
        _ensure_fake_registered()
        return Monitor(TimeFeatureWrapper(gym.make(job.env_id)))
    return Monitor(vhl._build_train_env(job.env_id, job.method_type, job.theta))


def _make_model(job, env):
    from stable_baselines3 import SAC, HerReplayBuffer
    return SAC(
        policy="MultiInputPolicy", env=env, seed=job.seed,
        gamma=vhl.TUNED_GAMMA, verbose=0,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future", copy_info_dict=True),
        buffer_size=min(1_000_000, job.training_steps),
        **_hparams(job),
    )


def _checkpoint_callback_class():
    from stable_baselines3.common.callbacks import BaseCallback

    class CheckpointCallback(BaseCallback):
        def __init__(self, store, every, start_steps):
            super().__init__()
            self.store, self.every = store, every
            self.last_saved = start_steps
            self.n_saved = 0

        def _on_step(self) -> bool:
            return True

        def _on_rollout_end(self) -> None:
            
            n = self.model.num_timesteps
            if n - self.last_saved < self.every:
                return
            rb = self.model.replay_buffer
            starts = getattr(rb, "_current_ep_start", None)
            if starts is None:
                raise RuntimeError("HerReplayBuffer has no _current_ep_start in this SB3 version; "
                                   "refusing to checkpoint (episode-boundary check impossible).")
            if not (starts == rb.pos).all():
                return                      
            try:
                slot = self.store.save(self.model, n)
                if self.n_saved == 0:
                    self.store.verify(slot)  
            except Exception as e:           
                if self.n_saved == 0:
                    raise RuntimeError(f"FIRST CHECKPOINT FAILED -- resumability is not working: {e!r}") from e
                print(f"  WARNING: checkpoint at step {n} failed ({e!r}); continuing without it.", flush=True)
                return
            self.n_saved += 1
            self.last_saved = n
            self.store.log(event="checkpoint", step=int(n), slot=slot)

    return CheckpointCallback


def _evaluate(job, model):
    import gymnasium as gym
    from sb3_contrib.common.wrappers import TimeFeatureWrapper
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
    return successes


def run_job(job: ResumableJob) -> dict:
    """Runs (or finishes, or resumes) one (label, seed) job. Safe to call repeatedly."""
    fp = job_fingerprint(job)
    store = CkptStore(job_dir(job), fp)
    store.check_fingerprint()

    done = _read_json(store.result_path())
    if done and done.get("fingerprint") == fp:
        return done

    env = _make_train_env(job)                      
    from stable_baselines3 import SAC
    tag = f"[{job.label} seed {job.seed}]"

    final_meta = _read_json(store.final_meta_path())
    final_ok = (final_meta and final_meta.get("fingerprint") == fp and os.path.exists(store.final_model_path())
                and os.path.getsize(store.final_model_path()) == final_meta.get("size"))
    if final_ok:
        store.log(event="resume_eval_only", step=int(final_meta["n_steps"]))
        print(f"{tag} training already complete; re-running evaluation only.", flush=True)
        model = SAC.load(store.final_model_path(), env=env, device="auto")
    else:
        cur = store.latest()
        n_resumes = sum(1 for e in store.events() if e.get("event") in ("resume", "resume_eval_only"))
        if cur is None:
            store.log(event="fresh_start")
            print(f"{tag} fresh start.", flush=True)
            model = _make_model(job, env)
            reset = True
        else:
            slot, meta = cur
            d = store.slot_dir(slot)
            model = SAC.load(os.path.join(d, "model.zip"), env=env, device="auto")
            model.load_replay_buffer(os.path.join(d, "buffer.pkl"))
            new_seed = (job.seed * 7919 + 104729 * (n_resumes + 1)) % (2 ** 31 - 1)
            model.set_random_seed(new_seed)
            store.log(event="resume", step=int(model.num_timesteps), slot=slot, reseed=int(new_seed))
            print(f"{tag} RESUMED from checkpoint at step {model.num_timesteps} ({slot}).", flush=True)
            reset = False

        cb_cls = _checkpoint_callback_class()
        cb = cb_cls(store, job.ckpt_every, start_steps=model.num_timesteps)
        remaining = job.training_steps - model.num_timesteps
        if reset:
            model.learn(total_timesteps=job.training_steps, callback=cb)
        elif remaining > 0:
            model.learn(total_timesteps=remaining, reset_num_timesteps=False, callback=cb)

        tmp = os.path.join(store.dir, "final_model.tmp.zip")
        model.save(tmp)
        _fsync(tmp)
        os.replace(tmp, store.final_model_path())
        _atomic_write_json(store.final_meta_path(), dict(
            fingerprint=fp, n_steps=int(model.num_timesteps), size=os.path.getsize(store.final_model_path())))
        store.log(event="training_complete", step=int(model.num_timesteps))

    successes = _evaluate(job, model)
    events = store.events()
    result = {
        "label": job.label, "seed": job.seed, "successes": successes,
        "n_eval_episodes": job.n_eval_episodes, "success_rate": successes / job.n_eval_episodes,
        "final_num_timesteps": int(model.num_timesteps),
        "n_resumes": sum(1 for e in events if e.get("event") == "resume"),
        "resume_steps": [e["step"] for e in events if e.get("event") == "resume"],
        "fingerprint": fp,
    }
    _atomic_write_json(store.result_path(), result)
    store.cleanup_slots()                           
    store.log(event="done", success_rate=result["success_rate"])
    return result





def selftest_resume():
    import multiprocessing as mp_
    tmp = tempfile.mkdtemp(prefix="resume_selftest_")
    hp = dict(batch_size=64, learning_starts=100, tau=0.05, learning_rate=1e-3,
              policy_kwargs=dict(net_arch=[32, 32], n_critics=2))
    job = ResumableJob(label="T", method_type="none", theta=None, seed=0, env_id=FAKE_ENV_ID,
                       training_steps=4000, n_eval_episodes=5, eval_seed_offset=1000,
                       ckpt_dir=tmp, ckpt_every=500, fake_env=True, hparam_override=hp)
    store = CkptStore(job_dir(job), job_fingerprint(job))
    ctx = mp_.get_context("spawn")
    try:
        print("1/6 starting a job in a child process and hard-killing it after >=3 checkpoints...")
        p = ctx.Process(target=run_job, args=(job,))
        p.start()
        t0 = time.time()
        while time.time() - t0 < 600:
            cur = store.latest()
            if cur and cur[1]["n_steps"] >= 1500 and all(store._valid_meta(s) for s in store.SLOTS):
                break
            if not p.is_alive():
                raise AssertionError("child exited before reaching the kill point")
            time.sleep(0.5)
        else:
            raise AssertionError("timed out waiting for checkpoints")
        p.kill()
        p.join()
        newest_slot, newest_meta = store.latest()
        other_slot = store.SLOTS[1 - store.SLOTS.index(newest_slot)]
        other_meta = store._valid_meta(other_slot)
        print(f"    killed; valid checkpoints: {newest_slot}@{newest_meta['n_steps']}, {other_slot}@{other_meta['n_steps']}")
        assert not os.path.exists(store.result_path())

        print("2/6 simulating a power cut DURING a checkpoint write: truncating the newest slot's model.zip...")
        mp_path = os.path.join(store.slot_dir(newest_slot), "model.zip")
        with open(mp_path, "rb") as f:
            blob = f.read()
        with open(mp_path, "wb") as f:
            f.write(blob[: len(blob) // 2])
        fallback = store.latest()
        assert fallback is not None and fallback[0] == other_slot and fallback[1]["n_steps"] == other_meta["n_steps"], fallback
        print(f"    loader correctly falls back to {fallback[0]}@{fallback[1]['n_steps']}")

        print("3/6 re-running the same job (should RESUME from the older valid checkpoint, not restart)...")
        result = run_job(job)
        ev = store.events()
        resumes = [e for e in ev if e["event"] == "resume"]
        assert len(resumes) == 1 and resumes[0]["step"] == other_meta["n_steps"], resumes
        assert result["final_num_timesteps"] == job.training_steps, result
        assert result["n_resumes"] == 1 and result["resume_steps"] == [other_meta["n_steps"]]
        assert os.path.exists(store.result_path())
        print(f"    resumed at {resumes[0]['step']}, finished at {result['final_num_timesteps']} steps, "
              f"success_rate={result['success_rate']:.0%}")

        print("4/6 re-running again (must be an instant no-op returning the stored result)...")
        t0 = time.time()
        again = run_job(job)
        assert again == result and time.time() - t0 < 5
        assert not os.path.exists(store.slot_dir("slot0")) and not os.path.exists(store.slot_dir("slot1"))

        print("5/6 config-mismatch guard (change training_steps -> must refuse)...")
        import dataclasses
        job2 = dataclasses.replace(job, training_steps=5000)
        try:
            run_job(job2)
            raise AssertionError("mismatch was NOT detected")
        except ConfigMismatch:
            print("    refused, as intended")

        print("6/6 eval-only resume (delete result.json, keep final model)...")
        os.remove(store.result_path())
        r3 = run_job(job)
        assert r3["final_num_timesteps"] == job.training_steps
        assert any(e["event"] == "resume_eval_only" for e in store.events())
        print("\nSELF-TEST PASSED: hard kill -> corrupted newest slot -> fallback -> resume -> finish -> no-op rerun -> "
              "mismatch guard -> eval-only resume.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)





def build_arg_parser():
    p = vhl.build_arg_parser()
    p.description = __doc__
    p.add_argument("--ckpt-dir", default="resumable_ckpt")
    p.add_argument("--ckpt-every", type=int, default=1000,
                   help="Checkpoint interval in env steps (taken at the next episode boundary).")
    return p


def run_validation(args):
    method_specs = vhl.load_method_specs(args.method)
    if args.validate_only:
        vhl.run_preflight_validation(method_specs, args.env_id)
        return

    jobs = []
    for label, method_type, theta in method_specs:
        for seed in range(args.seed_offset, args.seed_offset + args.n_seeds):
            jobs.append(ResumableJob(
                label=label, method_type=method_type, theta=theta, seed=seed, env_id=args.env_id,
                training_steps=args.training_steps, n_eval_episodes=args.n_eval_episodes,
                eval_seed_offset=args.eval_seed_offset, ckpt_dir=args.ckpt_dir, ckpt_every=args.ckpt_every))

    results_by_label = {label: [] for label, _, _ in method_specs}
    todo, n_resume = [], 0
    for j in jobs:
        store = CkptStore(job_dir(j), job_fingerprint(j))
        store.check_fingerprint()                          
        done = _read_json(store.result_path())
        if done and done.get("fingerprint") == store.fp:
            results_by_label[j.label].append(done)
            print(f"  [{j.label}] seed {j.seed}: already complete ({done['success_rate']:.0%}) -- skipping")
        else:
            cur = store.latest()
            n_resume += cur is not None
            todo.append(j)
            print(f"  [{j.label}] seed {j.seed}: " + (f"will RESUME from step {cur[1]['n_steps']}" if cur else "fresh start"))
    print(f"\n{len(jobs) - len(todo)} complete, {n_resume} to resume, {len(todo) - n_resume} fresh; "
          f"{args.training_steps} steps each; checkpoints every {args.ckpt_every} steps in '{args.ckpt_dir}'.\n")

    def dump(status):
        _atomic_write_json(args.output, {"status": status, "raw_results": results_by_label})

    dump("in_progress")
    t0 = time.time()
    if todo:
        with ProcessPoolExecutor(max_workers=args.max_workers, mp_context=mp.get_context("spawn"),
                                 initializer=vhl._worker_init) as pool:
            futures = {pool.submit(run_job, j): j for j in todo}
            for fut in as_completed(futures):
                j = futures[fut]
                try:
                    r = fut.result()
                except Exception as e:
                    raise RuntimeError(f"Job failed: label={j.label} seed={j.seed}") from e
                results_by_label[r["label"]].append(r)
                n_done = sum(len(v) for v in results_by_label.values())
                print(f"  [{r['label']}] seed {r['seed']}: success={r['success_rate']:.0%} "
                      f"({r['successes']}/{r['n_eval_episodes']}) resumes={r['n_resumes']}  ({n_done}/{len(jobs)} done)",
                      flush=True)
                dump("in_progress")

    print(f"\nFinished in {(time.time() - t0) / 3600:.2f} h this session.\n")
    summary = {}
    print("=" * 60)
    print(f"RESULTS at {args.training_steps} steps -- TUNED hyperparameters + HER")
    print("=" * 60)
    for label, method_type, _ in method_specs:
        rs = sorted(results_by_label[label], key=lambda r: r["seed"])
        rates = [r["success_rate"] for r in rs]
        mean_rate = sum(rates) / len(rates)
        resumed = {r["seed"]: r["resume_steps"] for r in rs if r.get("n_resumes")}
        print(f"\n{label} ({method_type}): per-seed {[f'{x:.0%}' for x in rates]}, mean {mean_rate:.1%}")
        if resumed:
            print(f"  seeds resumed from checkpoints (disclose in the paper): {resumed}")
        summary[label] = {"method_type": method_type, "per_seed_success_rate": rates,
                          "mean_success_rate": mean_rate, "raw": rs}
    _atomic_write_json(args.output, {"status": "complete", "training_steps": args.training_steps, "summary": summary})
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    if "--selftest-resume" in sys.argv:
        selftest_resume()
    else:
        a = build_arg_parser().parse_args()
        if a.smoke_test:
            a.training_steps, a.n_seeds = 2000, 1
            a.ckpt_every = 500
            a.ckpt_dir = "resumable_ckpt_smoke"
            print("SMOKE TEST MODE: training_steps=2000, n_seeds=1, ckpt_every=500\n")
        print(f"CPU cores available: {os.cpu_count()}")
        run_validation(a)
