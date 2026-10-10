"""Experiment grids of the extended study (reviewer round). All grids are fixed here before any run."""
from dataclasses import replace

import numpy as np

from .general import RewardSpec, SysParams, corrected_penalties
from .gexp import GJob

SEEDS = list(range(10))
TUNE_SEEDS = [100, 101]
TUNE_STEPS = 50_000
MAIN_STEPS = 100_000
CURVE_EVERY = 5_000

NOMINAL = SysParams()

H3 = dict(base_ms=(20.0, 30.0, 40.0), slope_ms=(10.0, 15.0, 20.0), drain=(0.7, 0.6, 0.5))
THREE_CLASSES = dict(class_probs=(0.3, 0.3, 0.4), limits_ms=(40.0, 70.0, 110.0), eval_weights=(10.0, 4.0, 1.0),
                     cloud_meet=(0.0, 0.2, 0.6))
SCENARIOS = {
    "hetero3": SysParams(**H3, background=0.8, background_mode="bernoulli", cloud_meet=(0.0, 0.5)),
    "hetero3_3c": SysParams(**H3, **THREE_CLASSES, background=0.8, background_mode="bernoulli"),
    "large4_3c": SysParams(base_ms=(20.0, 25.0, 30.0, 40.0), slope_ms=(10.0, 12.0, 15.0, 20.0),
                           drain=(0.7, 0.6, 0.6, 0.5), **THREE_CLASSES, background=1.4,
                           background_mode="bernoulli"),
}
SCENARIO_STEPS = {"hetero3": 100_000, "hetero3_3c": 100_000, "large4_3c": 200_000}

SHIFTS = {
    "slower tasks (+5 ms per queued task)": replace(NOMINAL, slope_ms=(20.0, 20.0)),
    "slower service (d = 0.55)": replace(NOMINAL, drain=(0.55, 0.55)),
    "more background (0.3 per step)": replace(NOMINAL, background=0.3),
    "random background (Bernoulli 0.2)": replace(NOMINAL, background_mode="bernoulli"),
    "cloud meets analytics limit w.p. 0.5": replace(NOMINAL, cloud_meet=(0.0, 0.5)),
}
FINETUNE_STEPS = 25_000


def rewards_for(sysp):
    w = tuple(sysp.eval_weights)
    mean = float(np.dot(sysp.class_probs, w))
    return {
        "weighted": RewardSpec(kind="weighted", penalties=w),
        "corrected": RewardSpec(kind="weighted", penalties=corrected_penalties(w, sysp.class_probs, mean)),
        "flat": RewardSpec(kind="flat", penalties=w),
    }


# hyperparameter search ------------------------------------------------------------------------
def _configs(algo, n=6, seed=0):
    rng = np.random.default_rng({"ppo": 11, "dqn": 12, "a2c": 13}[algo] + seed)
    space = {
        "ppo": dict(learning_rate=[1e-4, 3e-4, 1e-3], n_steps=[512, 1024, 2048], batch_size=[64, 128],
                    ent_coef=[0.0, 0.01], gamma=[0.99, 0.999], net_arch=[(64, 64), (128, 128)]),
        "dqn": dict(learning_rate=[1e-4, 3e-4, 1e-3], target_update_interval=[500, 2000],
                    exploration_fraction=[0.1, 0.3], buffer_size=[100_000, 1_000_000], gamma=[0.99, 0.999],
                    reward_scale=[1.0, 0.1], net_arch=[(64, 64), (128, 128)]),
        "a2c": dict(learning_rate=[3e-4, 7e-4, 1e-3], n_steps=[5, 32, 128], gamma=[0.99, 0.999],
                    ent_coef=[0.0, 0.01], normalize_advantage=[False, True], net_arch=[(64, 64), (128, 128)]),
    }[algo]
    out = [()]  # library default (DQN: the settings of the main study)
    while len(out) < n:
        cfg = tuple(sorted((k, v[int(rng.integers(len(v)))]) for k, v in space.items()))
        if cfg not in out:
            out.append(cfg)
    return out


def split_cfg(cfg):
    cfg = dict(cfg)
    scale = cfg.pop("reward_scale", 1.0)
    return tuple(sorted(cfg.items())), scale


def tune_jobs():
    jobs = {}
    rw = rewards_for(NOMINAL)
    for algo in ["ppo", "dqn", "a2c"]:
        for i, cfg in enumerate(_configs(algo)):
            kw, scale = split_cfg(cfg)
            for rname in ["weighted", "flat"]:
                r = replace(rw[rname], scale=scale)
                for s in TUNE_SEEDS:
                    jobs[(algo, i, rname, s)] = GJob(algo, NOMINAL, r, s, TUNE_STEPS, algo_kwargs=kw,
                                                     tag=f"tune {algo} cfg{i} {rname}")
    return jobs


def select_configs(load):
    """Best configuration per algorithm and reward: highest mean exact expected return under its own
    (unscaled) training reward over the tuning seeds. The SLA cost is not used for the choice."""
    jobs = tune_jobs()
    best = {}
    for algo in ["ppo", "dqn", "a2c"]:
        for rname in ["weighted", "flat"]:
            scores = []
            for i, cfg in enumerate(_configs(algo)):
                vals = [load(jobs[(algo, i, rname, s)]) for s in TUNE_SEEDS]
                if any(v is None for v in vals):
                    return None
                scores.append((float(np.mean([v["expected_return"] for v in vals])), i, cfg))
            sc, i, cfg = max(scores)
            best[(algo, rname)] = {"index": i, "config": cfg, "score": sc, "all": [(s, j) for s, j, _ in scores]}
    return best


def main_jobs(best):
    jobs = {}
    rw = rewards_for(NOMINAL)
    plan = {"ppo": ["weighted", "corrected", "flat"], "dqn": ["weighted", "flat"],
            "a2c": ["weighted", "flat"]}
    for algo, names in plan.items():
        for rname in names:
            src = "weighted" if rname == "corrected" else rname
            kw, scale = split_cfg(best[(algo, src)]["config"])
            r = replace(rw[rname], scale=scale)
            for s in SEEDS:
                jobs[(algo, rname, s)] = GJob(algo, NOMINAL, r, s, MAIN_STEPS, algo_kwargs=kw,
                                              curve_every=CURVE_EVERY, save_model=(algo == "ppo"),
                                              tag=f"main {algo} {rname}")
    return jobs


def scenario_jobs(best):
    jobs = {}
    for name, sp in SCENARIOS.items():
        for rname, r in rewards_for(sp).items():
            src = "weighted" if rname == "corrected" else rname
            kw, scale = split_cfg(best[("ppo", src)]["config"])
            for s in SEEDS:
                jobs[(name, rname, s)] = GJob("ppo", sp, replace(r, scale=scale), s, SCENARIO_STEPS[name],
                                              algo_kwargs=kw, curve_every=20_000, tag=f"scen {name} {rname}")
    return jobs


def published_jobs(best):
    """Tiered adaptive penalty with risk features and action pruning, after Yamsani and Chenna Reddy."""
    jobs = {}
    kw, _ = split_cfg(best[("ppo", "weighted")]["config"])
    plan = [("nominal", NOMINAL, "full", 0.001), ("nominal", NOMINAL, "full", 0.01),
            ("nominal", NOMINAL, "adaptive_only", 0.01)]
    for sname, sp, variant, eta in plan:
        w = tuple(sp.eval_weights)
        r = RewardSpec(kind="tiered_adaptive", penalties=w, target=tuple([0.05] * sp.k), eta=eta, window=200)
        full = variant == "full"
        for s in SEEDS:
            jobs[(sname, variant, eta, s)] = GJob("ppo", sp, r, s, MAIN_STEPS, algo_kwargs=kw, svrs=full,
                                                  prune=full, tag=f"pub {sname} {variant} eta{eta}")
    return jobs


def finetune_jobs(best, main):
    jobs = {}
    rw = rewards_for(NOMINAL)
    for shift, sp in SHIFTS.items():
        for rname in ["weighted", "flat"]:
            kw, scale = split_cfg(best[("ppo", rname)]["config"])
            r_sp = rewards_for(sp)[rname]
            for s in SEEDS:
                src = main[("ppo", rname, s)]
                jobs[(shift, rname, s)] = GJob("ppo", sp, replace(r_sp, scale=scale), s, FINETUNE_STEPS,
                                               algo_kwargs=kw, init_from=src.name() + ".zip",
                                               tag=f"ft {shift} {rname}")
    return jobs


# per-system tuning of PPO --------------------------------------------------------------------
# The three PPO configurations with the highest mean expected return over both rewards in the
# nominal tuning (configurations 4, 0 and 5), each trained at the full budget of the system with the
# tuning seeds. Fixed before any of these runs.
SCEN_TUNE_CFGS = [4, 0, 5]


def scen_tune_jobs():
    jobs = {}
    cfgs = _configs("ppo")
    for name, sp in SCENARIOS.items():
        rw = rewards_for(sp)
        for i in SCEN_TUNE_CFGS:
            kw, scale = split_cfg(cfgs[i])
            for rname in ["weighted", "flat"]:
                for s in TUNE_SEEDS:
                    jobs[(name, i, rname, s)] = GJob("ppo", sp, replace(rw[rname], scale=scale), s,
                                                     SCENARIO_STEPS[name], algo_kwargs=kw, curve_every=20_000,
                                                     tag=f"stune {name} cfg{i} {rname}")
    return jobs


def select_scen_configs(load):
    jobs = scen_tune_jobs()
    cfgs = _configs("ppo")
    best = {}
    for name in SCENARIOS:
        for rname in ["weighted", "flat"]:
            scores = []
            for i in SCEN_TUNE_CFGS:
                vals = [load(jobs[(name, i, rname, s)]) for s in TUNE_SEEDS]
                if any(v is None for v in vals):
                    return None
                scores.append((float(np.mean([v["expected_return"] for v in vals])), i))
            sc, i = max(scores)
            best[(name, rname)] = {"index": i, "config": cfgs[i], "score": sc, "all": scores}
    return best


def scenario_jobs_tuned(best_scen):
    jobs = {}
    for name, sp in SCENARIOS.items():
        for rname, r in rewards_for(sp).items():
            src = "weighted" if rname == "corrected" else rname
            kw, scale = split_cfg(best_scen[(name, src)]["config"])
            for s in SEEDS:
                jobs[(name, rname, s)] = GJob("ppo", sp, replace(r, scale=scale), s, SCENARIO_STEPS[name],
                                              algo_kwargs=kw, curve_every=20_000, tag=f"scen2 {name} {rname}")
    return jobs
