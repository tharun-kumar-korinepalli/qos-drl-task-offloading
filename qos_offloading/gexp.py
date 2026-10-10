"""Training jobs on the general system, with exact evaluation, caching and parallel execution."""
import hashlib
import json
import multiprocessing as mp
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .general import GeneralEnv, GeneralModel, RewardSpec, SysParams
from .runner import versions

ROOT = Path(__file__).resolve().parent.parent
GRUNS = Path(os.environ.get("QOS_GRUNS_DIR", ROOT / "results" / "gruns"))
GTABLES = ROOT / "results" / "gtables"
GMODELS = ROOT / "results" / "gmodels"


@dataclass(frozen=True)
class GJob:
    algo: str
    sysp: SysParams
    reward: RewardSpec
    seed: int
    timesteps: int
    algo_kwargs: tuple = ()
    curve_every: int = 0
    svrs: bool = False
    prune: bool = False
    save_model: bool = False
    init_from: str = ""
    tag: str = ""

    def spec(self):
        return {"algo": self.algo, "sys": self.sysp.to_dict(), "reward": self.reward.to_dict(), "seed": self.seed,
                "timesteps": self.timesteps, "algo_kwargs": dict(self.algo_kwargs), "curve_every": self.curve_every,
                "svrs": self.svrs, "prune": self.prune, "init_from": self.init_from}

    def key(self):
        return hashlib.sha1(json.dumps(self.spec(), sort_keys=True).encode()).hexdigest()[:16]

    def name(self):
        return f"{self.algo}_{self.key()}_s{self.seed}"

    def path(self):
        return GRUNS / f"{self.name()}.json"


def make_algo(job, env):
    from stable_baselines3 import A2C, DQN, PPO
    kw = dict(job.algo_kwargs)
    if "net_arch" in kw:
        kw.setdefault("policy_kwargs", {})
        kw["policy_kwargs"] = {**kw["policy_kwargs"], "net_arch": list(kw.pop("net_arch"))}
    if job.prune:
        from sb3_contrib import MaskablePPO
        return MaskablePPO("MlpPolicy", env, verbose=0, seed=job.seed, **kw)
    cls = {"ppo": PPO, "dqn": DQN, "a2c": A2C}[job.algo]
    if job.algo == "dqn":
        kw = {**dict(learning_starts=1000, target_update_interval=500, exploration_fraction=0.3,
                     exploration_final_eps=0.05), **kw}
    return cls("MlpPolicy", env, verbose=0, seed=job.seed, **kw)


def load_algo(job, path, env):
    from stable_baselines3 import A2C, DQN, PPO
    cls = {"ppo": PPO, "dqn": DQN, "a2c": A2C}[job.algo]
    return cls.load(path, env=env, seed=job.seed)


def table_of(model_rl, exact, prune_env=None):
    import torch
    if prune_env is not None:
        raise ValueError("masked policies are evaluated by simulation")

    def fn(obs):
        with torch.no_grad():
            return model_rl.predict(obs, deterministic=True)[0]
    return exact.policy_from_fn(fn)


def simulate(model_rl, job, episodes, seed0=10_000):
    """Monte Carlo evaluation in the simulator (used for agents whose policy is not a table)."""
    env = GeneralEnv(job.sysp, RewardSpec(kind="weighted", penalties=tuple(job.sysp.eval_weights)),
                     svrs_features=job.svrs, prune=job.prune)
    K = job.sysp.k
    viol = np.zeros(K)
    cnt = np.zeros(K)
    edge = 0
    per_ep = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        done = False
        v_ep = np.zeros(K)
        c_ep = np.zeros(K)
        while not done:
            if job.prune:
                a = int(model_rl.predict(obs, deterministic=True, action_masks=env.action_masks())[0])
            else:
                a = int(model_rl.predict(obs, deterministic=True)[0])
            obs, r, done, _, info = env.step(a)
            c = info["task_class"]
            c_ep[c] += 1
            v_ep[c] += info["violated"]
            edge += a < job.sysp.n
        viol += v_ep
        cnt += c_ep
        ew = np.array(job.sysp.eval_weights)
        p = np.array(job.sysp.class_probs)
        per_ep.append(float(p @ (ew * v_ep / np.maximum(c_ep, 1))))
    vr = viol / cnt
    ew = np.array(job.sysp.eval_weights)
    p = np.array(job.sysp.class_probs)
    return {"miss": vr.tolist(), "cost": float(p @ (ew * vr)), "edge": float(edge / cnt.sum()),
            "overall": float(viol.sum() / cnt.sum()), "episodes": episodes,
            "cost_se": float(np.std(per_ep, ddof=1) / np.sqrt(episodes))}


def execute(job):
    import torch
    torch.set_num_threads(1)
    from stable_baselines3.common.callbacks import BaseCallback

    out = job.path()
    if out.exists():
        return str(out), True
    t0 = time.time()
    env = GeneralEnv(job.sysp, job.reward, svrs_features=job.svrs, prune=job.prune)
    env.reset(seed=job.seed)
    exact = None if (job.svrs or job.prune) else GeneralModel(job.sysp)
    if job.init_from:
        model = load_algo(job, GMODELS / job.init_from, env)
    else:
        model = make_algo(job, env)

    curve = []

    class CB(BaseCallback):
        def _on_step(self):
            if job.curve_every and self.num_timesteps % job.curve_every == 0:
                if exact is not None:
                    m = exact.evaluate(table_of(self.model, exact))
                else:
                    m = simulate(self.model, job, 50)
                curve.append({"timesteps": int(self.num_timesteps), **m})
            return True

    model.learn(total_timesteps=job.timesteps, callback=CB(), reset_num_timesteps=not job.init_from)
    train_s = time.time() - t0
    rec = {"key": job.key(), "spec": job.spec(), "tag": job.tag, "train_seconds": train_s,
           "timesteps_done": int(model.num_timesteps), "curve": curve, "versions": versions()}
    if exact is not None:
        tab = table_of(model, exact)
        rec["exact"] = exact.evaluate(tab)
        rec["expected_return"] = exact.expected_return(tab, job.reward)
        GTABLES.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(GTABLES / f"{job.name()}.npz", table=tab)
        rec["mc"] = simulate(model, job, 100)
    else:
        rec["mc"] = simulate(model, job, 500)
    if job.save_model:
        GMODELS.mkdir(parents=True, exist_ok=True)
        model.save(GMODELS / f"{job.name()}.zip")
    if hasattr(env, "lam"):
        rec["final_penalties"] = [float(x) for x in env.lam]
    rec["wall_seconds"] = time.time() - t0
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1))
    tmp.replace(out)
    return str(out), False


def run(jobs, workers=2, log=print):
    uniq = list({j.path(): j for j in jobs}.values())
    todo = [j for j in uniq if not j.path().exists()]
    # longest jobs first
    todo.sort(key=lambda j: -j.timesteps * (j.sysp.n + 1))
    log(f"{len(uniq)} jobs, {len(uniq) - len(todo)} cached, {len(todo)} to run")
    if not todo:
        return
    t0 = time.time()
    if workers <= 1:
        for i, j in enumerate(todo):
            execute(j)
            log(f"[{i + 1}/{len(todo)}] {j.name()} {j.tag} ({time.time() - t0:.0f}s)")
        return
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, maxtasksperchild=2) as pool:
        for i, (p, _) in enumerate(pool.imap_unordered(execute, todo)):
            log(f"[{i + 1}/{len(todo)}] {Path(p).name} ({time.time() - t0:.0f}s)")


def load(job):
    p = job.path()
    return json.loads(p.read_text()) if p.exists() else None
