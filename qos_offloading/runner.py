import hashlib
import json
import multiprocessing as mp
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .params import EnvParams

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = Path(os.environ.get("QOS_RUNS_DIR", ROOT / "results" / "runs"))


@dataclass(frozen=True)
class Job:
    algo: str
    train: EnvParams
    seed: int
    timesteps: int
    eval: EnvParams = None
    algo_kwargs: tuple = ()
    curve_every: int = 0

    def spec(self):
        ev = (self.eval or self.train).eval_view()
        return {
            "algo": self.algo,
            "train": self.train.canonical().to_dict(),
            "eval": ev.to_dict(),
            "seed": int(self.seed),
            "timesteps": int(self.timesteps),
            "algo_kwargs": dict(self.algo_kwargs),
            "curve_every": int(self.curve_every),
        }

    def key(self):
        blob = json.dumps(self.spec(), sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:16]

    def path(self):
        return RUNS_DIR / f"{self.algo}_{self.key()}_s{self.seed}.json"


def versions():
    import gymnasium
    import numpy
    import stable_baselines3
    import torch
    return {"python": sys.version.split()[0], "numpy": numpy.__version__, "gymnasium": gymnasium.__version__,
            "stable_baselines3": stable_baselines3.__version__, "torch": torch.__version__,
            "platform": platform.platform()}


def execute(job):
    import torch
    torch.set_num_threads(1)
    from .agents import train_and_evaluate

    out = job.path()
    if out.exists():
        return str(out), True
    t0 = time.time()
    res = train_and_evaluate(job.algo, job.train, job.seed, job.timesteps, eval_params=job.eval,
                             algo_kwargs=dict(job.algo_kwargs), curve_every=job.curve_every or None)
    record = {"key": job.key(), "spec": job.spec(), **res, "wall_seconds": time.time() - t0,
              "versions": versions()}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=1))
    tmp.replace(out)
    return str(out), False


def run_jobs(jobs, workers=1, log=print):
    unique = list({j.path(): j for j in jobs}.values())
    todo = [j for j in unique if not j.path().exists()]
    log(f"{len(unique)} unique jobs, {len(unique) - len(todo)} cached, {len(todo)} to run, {workers} workers")
    if not todo:
        return
    t0 = time.time()
    done = 0
    if workers <= 1:
        it = map(execute, todo)
        for path, _ in it:
            done += 1
            log(f"[{done}/{len(todo)}] {Path(path).name} ({time.time() - t0:.0f}s)")
        return
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, maxtasksperchild=4) as pool:
        for path, _ in pool.imap_unordered(execute, todo):
            done += 1
            log(f"[{done}/{len(todo)}] {Path(path).name} ({time.time() - t0:.0f}s)")


def load(job):
    p = job.path()
    return json.loads(p.read_text()) if p.exists() else None
