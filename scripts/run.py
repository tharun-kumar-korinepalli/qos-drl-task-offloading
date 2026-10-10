import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qos_offloading import study as S
from qos_offloading.training import execute, load

STAGES = ["tune", "main", "systems_nominal_setting", "system_tune", "systems", "shifts", "published"]


def run_stage(jobs, shard, nshard):
    jobs = sorted({j.path(): j for j in jobs}.values(), key=lambda j: (-j.timesteps * (j.system.n + 1), j.name()))
    mine = [j for i, j in enumerate(jobs) if i % nshard == shard]
    t0 = time.time()
    for i, j in enumerate(mine):
        if not j.path().exists():
            execute(j)
            print(f"[{shard}] {i + 1}/{len(mine)} {j.tag} ({time.time() - t0:.0f}s)", flush=True)
    while not all(j.path().exists() for j in jobs):
        time.sleep(20)


def wait(select):
    while (best := select(load)) is None:
        time.sleep(20)
    return best


def main(stages, shard, nshard):
    if "tune" in stages:
        run_stage(S.tune_jobs().values(), shard, nshard)
    best = wait(S.select_configs)
    main_jobs = S.main_jobs(best)
    if "main" in stages:
        run_stage(main_jobs.values(), shard, nshard)
    if "systems_nominal_setting" in stages:
        run_stage(S.system_jobs_nominal_setting(best).values(), shard, nshard)
    if "system_tune" in stages:
        run_stage(S.system_tune_jobs().values(), shard, nshard)
    if "systems" in stages:
        run_stage(S.system_jobs(wait(S.select_system_configs)).values(), shard, nshard)
    if "shifts" in stages:
        run_stage(S.shift_jobs(best, main_jobs).values(), shard, nshard)
    if "published" in stages:
        run_stage(S.published_jobs(best).values(), shard, nshard)


if __name__ == "__main__":
    stages = sys.argv[1].split(",") if len(sys.argv) > 1 and sys.argv[1] != "all" else STAGES
    shard, nshard = (int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "0/1").split("/"))
    main(stages, shard, nshard)
