"""Wall-clock cost of the exact DP against training one PPO agent (reviewer: runtime of training vs. DP solve).

    python scripts/runtimes.py            # all four systems
    python scripts/runtimes.py large4_3c  # one system

Writes results/summary/runtimes.json. Each PPO agent is trained once with seed 0, the tuned setting of the
weighted reward and the main budget of its system, without learning curves, single-threaded like the main runs.
"""
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from qos_offloading import ExactModel, OffloadingEnv
from qos_offloading import study as S
from qos_offloading.training import load, make_agent, policy_table

OUT = ROOT / "results" / "summary" / "runtimes.json"


def cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main(names):
    import torch
    torch.set_num_threads(1)
    best = S.select_configs(load)
    best_sys = S.select_system_configs(load)
    systems = {"nominal": S.NOMINAL, **S.SYSTEMS}
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    out["cpu"] = cpu_model()
    out["platform"] = platform.platform()
    for name in names:
        sp = systems[name]
        t0 = time.time()
        model = ExactModel(sp)
        build = time.time() - t0
        t1 = time.time()
        pol, _ = model.solve(model.cost_eval())
        solve = time.time() - t1
        t2 = time.time()
        model.evaluate(pol)
        evaluate = time.time() - t2

        cfg = best[("ppo", "weighted")]["config"] if name == "nominal" else best_sys[(name, "weighted")]["config"]
        steps = S.MAIN_STEPS if name == "nominal" else S.SYSTEM_STEPS[name]
        job = S._job("ppo", sp, S.rewards_for(sp)["weighted"], cfg, 0, steps, tag=f"runtime {name}")
        env = OffloadingEnv(sp, job.reward)
        env.reset(seed=0)
        agent = make_agent(job, env)
        t3 = time.time()
        agent.learn(total_timesteps=steps)
        train = time.time() - t3
        t4 = time.time()
        table = policy_table(agent, model)
        to_table = time.time() - t4
        cost = model.evaluate(table)["cost"]
        out[name] = {"states": int(model.S * model.K * model.T), "build_s": build, "solve_s": solve,
                     "evaluate_s": evaluate, "ppo_steps": int(agent.num_timesteps), "ppo_train_s": train,
                     "policy_table_s": to_table, "ppo_cost": cost}
        print(name, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in out[name].items()}, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:] or ["nominal", "hetero3", "hetero3_3c", "large4_3c"])
