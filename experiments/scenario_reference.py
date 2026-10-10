"""Exact references for every system: rules, tuned rules, optimum, DP policy of each reward, frontier,
discount and time-invariance checks, and DP size and time."""
import json
import time
from pathlib import Path

import numpy as np

from qos_offloading.general import GeneralModel, RewardSpec, reference
from qos_offloading import studies_v3 as V

OUT = Path(__file__).resolve().parent.parent / "results" / "summary" / "scenario_reference.json"


def frontier(model, n=101):
    pts = []
    p = model.p
    for lam in np.linspace(0.0, 1.0, n):
        w = (lam / p[0], (1 - lam) / p[1])
        pol, _ = model.solve(model.cost_eval(w))
        m = model.evaluate(pol)
        pts.append((float(lam), m["miss"][0], m["miss"][1]))
    return pts


def main():
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    systems = {"nominal": V.NOMINAL, **V.SCENARIOS}
    for name, sp in systems.items():
        if name in out:
            continue
        t0 = time.time()
        model, ref = reference(sp, sets=True)
        th = ref.pop("_reservation_th")
        ref["build_and_rules_seconds"] = time.time() - t0
        t1 = time.time()
        pol, _ = model.solve(model.cost_eval())
        ref["solve_seconds"] = time.time() - t1
        mid = np.broadcast_to(pol[model.T // 2], pol.shape).copy()
        ref["optimum_time_invariant"] = model.evaluate(mid)
        ref["dp_rewards"] = {}
        for rname, r in V.rewards_for(sp).items():
            rp, _ = model.solve(model.cost_reward(r))
            ref["dp_rewards"][rname] = {**model.evaluate(rp), "penalties": list(r.effective_penalties(sp))}
            rg, _ = model.solve(model.cost_reward(r), gamma=0.99)
            ref["dp_rewards"][rname + "_gamma0.99"] = model.evaluate(rg)
        if sp.k == 2:
            ref["frontier"] = frontier(model)
        ref["states"] = int(model.S * model.K * model.T)
        out[name] = ref
        OUT.write_text(json.dumps(out, indent=1))
        print(name, f"{time.time() - t0:.0f}s", {k: round(v["cost"], 4) for k, v in ref.items()
                                                  if isinstance(v, dict) and "cost" in v},
              {k: round(v["cost"], 4) for k, v in ref["dp_rewards"].items()}, flush=True)


if __name__ == "__main__":
    main()
