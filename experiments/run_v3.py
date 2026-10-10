"""Runs the extended study: tuning, main study at 100k steps, larger systems, published reward designs,
fine-tuning under model shift. Every stage is cached, so the script can be stopped and restarted."""
import json
import sys
import time
from pathlib import Path

from qos_offloading import gexp
from qos_offloading import studies_v3 as V

ROOT = Path(__file__).resolve().parent.parent
BEST = ROOT / "results" / "summary" / "tuning_selection.json"


def log(msg):
    print(msg, flush=True)


def best_configs():
    best = V.select_configs(gexp.load)
    if best is None:
        raise SystemExit("tuning incomplete")
    BEST.parent.mkdir(parents=True, exist_ok=True)
    BEST.write_text(json.dumps({f"{a}|{r}": {**v, "config": [list(x) for x in v["config"]]}
                                for (a, r), v in best.items()}, indent=1))
    return best


def run_stage(jobs, shard, nshard):
    """Each shard trains its share of the jobs, then waits until every job of the stage exists."""
    jobs = sorted({j.path(): j for j in jobs}.values(), key=lambda j: (-j.timesteps * (j.sysp.n + 1), j.name()))
    mine = [j for i, j in enumerate(jobs) if i % nshard == shard]
    t0 = time.time()
    for i, j in enumerate(mine):
        if not j.path().exists():
            gexp.execute(j)
            log(f"[shard {shard}] {i + 1}/{len(mine)} {j.name()} {j.tag} ({time.time() - t0:.0f}s)")
    while not all(j.path().exists() for j in jobs):
        time.sleep(20)


def wait_best():
    while True:
        best = V.select_configs(gexp.load)
        if best is not None:
            return best
        time.sleep(20)


def main(stages, shard, nshard):
    if "tune" in stages:
        log("== tuning")
        run_stage(list(V.tune_jobs().values()), shard, nshard)
    best = best_configs() if shard == 0 else wait_best()
    for k, v in best.items():
        log(f"selected {k}: cfg{v['index']} {v['config']} return {v['score']:.1f}")
    main_j = V.main_jobs(best)
    if "main" in stages:
        log("== main")
        run_stage(list(main_j.values()), shard, nshard)
    if "scen" in stages:
        log("== scenarios")
        run_stage(list(V.scenario_jobs(best).values()), shard, nshard)
    if "stune" in stages:
        log("== scenario tuning")
        run_stage(list(V.scen_tune_jobs().values()), shard, nshard)
    if "scen2" in stages:
        while V.select_scen_configs(gexp.load) is None:
            time.sleep(20)
        bs = V.select_scen_configs(gexp.load)
        for k, v in bs.items():
            log(f"scenario selected {k}: cfg{v['index']} return {v['score']:.1f}")
        log("== scenarios, tuned per system")
        run_stage(list(V.scenario_jobs_tuned(bs).values()), shard, nshard)
    if "ft" in stages:
        log("== finetune")
        run_stage(list(V.finetune_jobs(best, main_j).values()), shard, nshard)

    if "pub" in stages:
        log("== published")
        run_stage(list(V.published_jobs(best).values()), shard, nshard)

if __name__ == "__main__":
    stages = sys.argv[1].split(",") if len(sys.argv) > 1 else ["tune", "main", "scen", "pub", "ft"]
    shard, nshard = (int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "0/1").split("/"))
    main(stages, shard, nshard)
