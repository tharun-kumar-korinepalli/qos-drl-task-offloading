from dataclasses import replace

import numpy as np

from .system import Reward, System, corrected_penalties
from .training import Job

SEEDS = list(range(10))
TUNE_SEEDS = [100, 101]
TUNE_STEPS = 50_000
MAIN_STEPS = 100_000
CURVE_EVERY = 5_000
SHIFT_STEPS = 25_000

NOMINAL = System()
_HETERO3 = dict(base_ms=(20.0, 30.0, 40.0), slope_ms=(10.0, 15.0, 20.0), drain=(0.7, 0.6, 0.5))
_THREE_CLASSES = dict(class_probs=(0.3, 0.3, 0.4), limits_ms=(40.0, 70.0, 110.0), eval_weights=(10.0, 4.0, 1.0),
                      cloud_meet=(0.0, 0.2, 0.6))
SYSTEMS = {
    "hetero3": System(**_HETERO3, background=0.8, background_mode="bernoulli", cloud_meet=(0.0, 0.5)),
    "hetero3_3c": System(**_HETERO3, **_THREE_CLASSES, background=0.8, background_mode="bernoulli"),
    "large4_3c": System(base_ms=(20.0, 25.0, 30.0, 40.0), slope_ms=(10.0, 12.0, 15.0, 20.0),
                        drain=(0.7, 0.6, 0.6, 0.5), **_THREE_CLASSES, background=1.4, background_mode="bernoulli"),
}
SYSTEM_STEPS = {"hetero3": 100_000, "hetero3_3c": 100_000, "large4_3c": 200_000}
SHIFTS = {
    "slower tasks (+5 ms per queued task)": replace(NOMINAL, slope_ms=(20.0, 20.0)),
    "slower service (d = 0.55)": replace(NOMINAL, drain=(0.55, 0.55)),
    "more background (0.3 per step)": replace(NOMINAL, background=0.3),
    "random background (Bernoulli 0.2)": replace(NOMINAL, background_mode="bernoulli"),
    "cloud meets analytics limit w.p. 0.5": replace(NOMINAL, cloud_meet=(0.0, 0.5)),
}
SYSTEM_TUNE_CONFIGS = [4, 0, 5]

_SPACE = {
    "ppo": dict(learning_rate=[1e-4, 3e-4, 1e-3], n_steps=[512, 1024, 2048], batch_size=[64, 128],
                ent_coef=[0.0, 0.01], gamma=[0.99, 0.999], net_arch=[(64, 64), (128, 128)]),
    "dqn": dict(learning_rate=[1e-4, 3e-4, 1e-3], target_update_interval=[500, 2000],
                exploration_fraction=[0.1, 0.3], buffer_size=[100_000, 1_000_000], gamma=[0.99, 0.999],
                reward_scale=[1.0, 0.1], net_arch=[(64, 64), (128, 128)]),
    "a2c": dict(learning_rate=[3e-4, 7e-4, 1e-3], n_steps=[5, 32, 128], gamma=[0.99, 0.999],
                ent_coef=[0.0, 0.01], normalize_advantage=[False, True], net_arch=[(64, 64), (128, 128)]),
}


def rewards_for(system):
    w = tuple(system.eval_weights)
    mean = float(np.dot(system.class_probs, w))
    return {"weighted": Reward(kind="weighted", penalties=w),
            "corrected": Reward(kind="weighted", penalties=corrected_penalties(w, system.class_probs, mean)),
            "flat": Reward(kind="flat", penalties=w)}


def configs(algo, n=6):
    rng = np.random.default_rng({"ppo": 11, "dqn": 12, "a2c": 13}[algo])
    out = [()]
    while len(out) < n:
        cfg = tuple(sorted((k, v[int(rng.integers(len(v)))]) for k, v in _SPACE[algo].items()))
        if cfg not in out:
            out.append(cfg)
    return out


def split(cfg):
    cfg = dict(cfg)
    scale = cfg.pop("reward_scale", 1.0)
    return tuple(sorted(cfg.items())), scale


def _job(algo, system, reward, cfg, seed, steps, **kw):
    algo_kwargs, scale = split(cfg)
    return Job(algo, system, replace(reward, scale=scale), seed, steps, algo_kwargs=algo_kwargs, **kw)


def tune_jobs():
    rw = rewards_for(NOMINAL)
    return {(algo, i, r, s): _job(algo, NOMINAL, rw[r], cfg, s, TUNE_STEPS, tag=f"tune {algo} cfg{i} {r}")
            for algo in ["ppo", "dqn", "a2c"] for i, cfg in enumerate(configs(algo))
            for r in ["weighted", "flat"] for s in TUNE_SEEDS}


def _select(jobs, keys, load):
    scores = []
    for i, cfg, k in keys:
        vals = [load(jobs[k + (s,)]) for s in TUNE_SEEDS]
        if any(v is None for v in vals):
            return None
        scores.append((float(np.mean([v["expected_return"] for v in vals])), i, cfg))
    score, i, cfg = max(scores)
    return {"index": i, "config": cfg, "score": score, "all": [(s, j) for s, j, _ in scores]}


def select_configs(load):
    jobs, best = tune_jobs(), {}
    for algo in ["ppo", "dqn", "a2c"]:
        for r in ["weighted", "flat"]:
            best[(algo, r)] = _select(jobs, [(i, c, (algo, i, r)) for i, c in enumerate(configs(algo))], load)
            if best[(algo, r)] is None:
                return None
    return best


def main_jobs(best):
    rw = rewards_for(NOMINAL)
    plan = {"ppo": ["weighted", "corrected", "flat"], "dqn": ["weighted", "flat"], "a2c": ["weighted", "flat"]}
    return {(algo, r, s): _job(algo, NOMINAL, rw[r], best[(algo, "flat" if r == "flat" else "weighted")]["config"],
                               s, MAIN_STEPS, curve_every=CURVE_EVERY, save_model=algo == "ppo",
                               tag=f"main {algo} {r}")
            for algo, names in plan.items() for r in names for s in SEEDS}


def system_jobs_nominal_setting(best):
    return {(name, r, s): _job("ppo", sp, rw, best[("ppo", "flat" if r == "flat" else "weighted")]["config"], s,
                               SYSTEM_STEPS[name], curve_every=20_000, tag=f"system nominal setting {name} {r}")
            for name, sp in SYSTEMS.items() for r, rw in rewards_for(sp).items() for s in SEEDS}


def system_tune_jobs():
    cfgs = configs("ppo")
    return {(name, i, r, s): _job("ppo", sp, rewards_for(sp)[r], cfgs[i], s, SYSTEM_STEPS[name], curve_every=20_000,
                                  tag=f"system tune {name} cfg{i} {r}")
            for name, sp in SYSTEMS.items() for i in SYSTEM_TUNE_CONFIGS for r in ["weighted", "flat"]
            for s in TUNE_SEEDS}


def select_system_configs(load):
    jobs, cfgs, best = system_tune_jobs(), configs("ppo"), {}
    for name in SYSTEMS:
        for r in ["weighted", "flat"]:
            best[(name, r)] = _select(jobs, [(i, cfgs[i], (name, i, r)) for i in SYSTEM_TUNE_CONFIGS], load)
            if best[(name, r)] is None:
                return None
    return best


def system_jobs(best_sys):
    return {(name, r, s): _job("ppo", sp, rw, best_sys[(name, "flat" if r == "flat" else "weighted")]["config"], s,
                               SYSTEM_STEPS[name], curve_every=20_000, tag=f"system {name} {r}")
            for name, sp in SYSTEMS.items() for r, rw in rewards_for(sp).items() for s in SEEDS}


def published_jobs(best):
    cfg = best[("ppo", "weighted")]["config"]
    w = tuple(NOMINAL.eval_weights)
    jobs = {}
    for variant, eta in [("full", 0.001), ("full", 0.01), ("adaptive_only", 0.01)]:
        rw = Reward(kind="tiered_adaptive", penalties=w, target=(0.05,) * NOMINAL.k, eta=eta, window=200)
        full = variant == "full"
        for s in SEEDS:
            jobs[("nominal", variant, eta, s)] = _job("ppo", NOMINAL, rw, cfg, s, MAIN_STEPS, risk_features=full,
                                                      prune=full, tag=f"published {variant} eta{eta}")
    return jobs


def shift_jobs(best, main):
    return {(shift, r, s): _job("ppo", sp, rewards_for(sp)[r], best[("ppo", r)]["config"], s, SHIFT_STEPS,
                                init_from=main[("ppo", r, s)].name() + ".zip", tag=f"shift {shift} {r}")
            for shift, sp in SHIFTS.items() for r in ["weighted", "flat"] for s in SEEDS}
