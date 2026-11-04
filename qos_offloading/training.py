import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .exact import ExactModel
from .system import OffloadingEnv, Reward, System

ROOT = Path(__file__).resolve().parent.parent
RUNS = Path(os.environ.get("QOS_RUNS_DIR", ROOT / "results" / "runs"))
TABLES = ROOT / "results" / "tables"
MODELS = ROOT / "results" / "models"
DQN_BASE = dict(learning_starts=1000, target_update_interval=500, exploration_fraction=0.3, exploration_final_eps=0.05)


@dataclass(frozen=True)
class Job:
    algo: str
    system: System
    reward: Reward
    seed: int
    timesteps: int
    algo_kwargs: tuple = ()
    curve_every: int = 0
    risk_features: bool = False
    prune: bool = False
    save_model: bool = False
    init_from: str = ""
    tag: str = ""

    def spec(self):
        return {"algo": self.algo, "sys": self.system.to_dict(), "reward": self.reward.to_dict(), "seed": self.seed,
                "timesteps": self.timesteps, "algo_kwargs": dict(self.algo_kwargs), "curve_every": self.curve_every,
                "svrs": self.risk_features, "prune": self.prune, "init_from": self.init_from}

    def key(self):
        return hashlib.sha1(json.dumps(self.spec(), sort_keys=True).encode()).hexdigest()[:16]

    def name(self):
        return f"{self.algo}_{self.key()}_s{self.seed}"

    def path(self):
        return RUNS / f"{self.name()}.json"


def load(job):
    p = job.path()
    return json.loads(p.read_text()) if p.exists() else None


def versions():
    import gymnasium
    import stable_baselines3
    import torch
    return {"python": sys.version.split()[0], "numpy": np.__version__, "gymnasium": gymnasium.__version__,
            "stable_baselines3": stable_baselines3.__version__, "torch": torch.__version__,
            "platform": platform.platform()}


def make_agent(job, env):
    from stable_baselines3 import A2C, DQN, PPO
    kw = dict(job.algo_kwargs)
    if "net_arch" in kw:
        kw["policy_kwargs"] = {"net_arch": list(kw.pop("net_arch"))}
    if job.prune:
        from sb3_contrib import MaskablePPO
        return MaskablePPO("MlpPolicy", env, verbose=0, seed=job.seed, **kw)
    if job.algo == "dqn":
        kw = {**DQN_BASE, **kw}
    return {"ppo": PPO, "dqn": DQN, "a2c": A2C}[job.algo]("MlpPolicy", env, verbose=0, seed=job.seed, **kw)


def load_agent(job, path, env):
    from stable_baselines3 import A2C, DQN, PPO
    return {"ppo": PPO, "dqn": DQN, "a2c": A2C}[job.algo].load(path, env=env, seed=job.seed)


def policy_table(agent, exact):
    import torch

    def fn(obs):
        with torch.no_grad():
            return agent.predict(obs, deterministic=True)[0]
    return exact.policy_from_fn(fn)


def simulate(agent, job, episodes, seed0=10_000):
    sp = job.system
    env = OffloadingEnv(sp, Reward(penalties=tuple(sp.eval_weights)), risk_features=job.risk_features, prune=job.prune)
    ew, p = np.array(sp.eval_weights), np.array(sp.class_probs)
    viol, cnt, edge, per_ep = np.zeros(sp.k), np.zeros(sp.k), 0, []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        done = False
        v_ep, c_ep = np.zeros(sp.k), np.zeros(sp.k)
        while not done:
            if job.prune:
                a = int(agent.predict(obs, deterministic=True, action_masks=env.action_masks())[0])
            else:
                a = int(agent.predict(obs, deterministic=True)[0])
            obs, _, done, _, info = env.step(a)
            c_ep[info["task_class"]] += 1
            v_ep[info["task_class"]] += info["violated"]
            edge += a < sp.n
        viol += v_ep
        cnt += c_ep
        per_ep.append(float(p @ (ew * v_ep / np.maximum(c_ep, 1))))
    vr = viol / cnt
    return {"miss": vr.tolist(), "cost": float(p @ (ew * vr)), "edge": float(edge / cnt.sum()),
            "overall": float(viol.sum() / cnt.sum()), "episodes": episodes,
            "cost_se": float(np.std(per_ep, ddof=1) / np.sqrt(episodes))}


def execute(job):
    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    torch.set_num_threads(1)

    out = job.path()
    if out.exists():
        return out
    t0 = time.time()
    env = OffloadingEnv(job.system, job.reward, risk_features=job.risk_features, prune=job.prune)
    env.reset(seed=job.seed)
    exact = None if (job.risk_features or job.prune) else ExactModel(job.system)
    agent = load_agent(job, MODELS / job.init_from, env) if job.init_from else make_agent(job, env)
    curve = []
    curve_time = [0.0]
    duals = []

    class Curve(BaseCallback):
        def _on_step(self):
            if job.curve_every and self.num_timesteps % job.curve_every == 0:
                tc = time.time()
                m = exact.evaluate(policy_table(self.model, exact)) if exact else simulate(self.model, job, 50)
                curve.append({"timesteps": int(self.num_timesteps), **m})
                if job.reward.kind == "lagrangian":
                    curve[-1]["multipliers"] = [float(x) for x in env.lam]
                curve_time[0] += time.time() - tc
            return True

        def _on_rollout_end(self):
            # PPO-Lagrangian: one dual step per collected rollout block (Sec. IV, Constrained Baseline)
            if job.reward.kind == "lagrangian":
                v_hat, lam = env.lagrange_update()
                duals.append({"timesteps": int(self.num_timesteps), "v_hat": v_hat, "multipliers": lam})

    t_learn = time.time()
    agent.learn(total_timesteps=job.timesteps, callback=Curve(), reset_num_timesteps=not job.init_from)
    learn_wall = time.time() - t_learn
    rec = {"key": job.key(), "spec": job.spec(), "tag": job.tag, "train_seconds": time.time() - t0,
           "learn_seconds_without_curve": learn_wall - curve_time[0], "curve_seconds": curve_time[0],
           "timesteps_done": int(agent.num_timesteps), "curve": curve, "versions": versions()}
    if duals:
        rec["duals"] = duals
    if exact is not None:
        table = policy_table(agent, exact)
        rec["exact"] = exact.evaluate(table)
        rec["expected_return"] = exact.expected_return(table, job.reward)
        TABLES.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(TABLES / f"{job.name()}.npz", table=table)
        rec["mc"] = simulate(agent, job, 100)
    else:
        rec["mc"] = simulate(agent, job, 500)
    if job.save_model:
        MODELS.mkdir(parents=True, exist_ok=True)
        agent.save(MODELS / f"{job.name()}.zip")
    if job.reward.kind in ("tiered_adaptive", "lagrangian"):
        rec["final_penalties"] = [float(x) for x in env.lam]
    rec["wall_seconds"] = time.time() - t0
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1))
    tmp.replace(out)
    return out
