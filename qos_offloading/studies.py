from dataclasses import replace

from . import config as C
from .params import EnvParams
from .runner import Job

BASE = EnvParams()
SEEDS = C.JOURNAL_SEEDS
T = C.TOTAL_TIMESTEPS

PENALTY_PAIRS = [(6.0, 6.0), (8.0, 4.0), (9.0, 3.0), (10.0, 2.0), (11.0, 1.0)]
DRAIN_PROBS = [0.9, 0.8, 0.7, 0.6, 0.5]
CRITICAL_PROBS = [0.3, 0.5, 0.7]
LEARNING_RATES = [1e-4, 3e-4, 1e-3]
ALGOS = ["ppo", "dqn", "a2c"]


def rewards(params):
    return {"weighted": params, "flat": replace(params, flat=True)}


def main_jobs():
    return {(algo, rw, s): Job(algo, p, s, T)
            for algo in ALGOS for rw, p in rewards(BASE).items() for s in SEEDS}


def penalty_jobs():
    return {(wc, wa, s): Job("ppo", replace(BASE, critical_penalty=wc, analytics_penalty=wa), s, T,
                             eval=BASE)
            for wc, wa in PENALTY_PAIRS for s in SEEDS}


def load_jobs():
    out = {}
    for d in DRAIN_PROBS:
        base = replace(BASE, drain_prob=d)
        for rw, p in rewards(base).items():
            for s in SEEDS:
                out[(d, rw, s)] = Job("ppo", p, s, T)
    return out


def mix_jobs():
    out = {}
    for pc in CRITICAL_PROBS:
        base = replace(BASE, critical_prob=pc)
        for rw, p in rewards(base).items():
            for s in SEEDS:
                out[(pc, rw, s)] = Job("ppo", p, s, T)
    return out


def curve_jobs():
    return {(algo, rw, s): Job(algo, p, s, C.CURVE_TIMESTEPS, curve_every=C.CURVE_EVERY)
            for algo in ["ppo", "dqn"] for rw, p in rewards(BASE).items() for s in C.CURVE_SEEDS}


def dqn_scaled_jobs():
    return {(rw, s): Job("dqn", replace(p, reward_scale=0.1), s, C.CURVE_TIMESTEPS, curve_every=C.CURVE_EVERY)
            for rw, p in rewards(BASE).items() for s in C.CURVE_SEEDS}


def lr_jobs():
    out = {}
    for lr in LEARNING_RATES:
        kw = () if lr == 3e-4 else (("learning_rate", lr),)
        for rw, p in rewards(BASE).items():
            for s in C.CURVE_SEEDS:
                out[(lr, rw, s)] = Job("ppo", p, s, T, algo_kwargs=kw)
    return out


STUDIES = {
    "main": main_jobs,
    "penalty": penalty_jobs,
    "load": load_jobs,
    "mix": mix_jobs,
    "lr": lr_jobs,
    "curves": curve_jobs,
    "dqn_scaled": dqn_scaled_jobs,
}
