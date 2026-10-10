import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from qos_offloading import reference
from qos_offloading import study as S

OUT = ROOT / "results" / "summary" / "references.json"


def main():
    out = {}
    for name, sp in {"nominal": S.NOMINAL, **S.SYSTEMS}.items():
        t0 = time.time()
        model, ref, _ = reference(sp, node_sets=True)
        pol, _ = model.solve(model.cost_eval())
        ref["optimum_time_invariant"] = model.evaluate(np.broadcast_to(pol[model.T // 2], pol.shape).copy())
        ref["dp_rewards"] = {}
        for r, rw in S.rewards_for(sp).items():
            ref["dp_rewards"][r] = {**model.evaluate(model.solve(model.cost_reward(rw))[0]),
                                    "penalties": list(rw.effective_penalties(sp))}
            ref["dp_rewards"][r + "_gamma0.99"] = model.evaluate(model.solve(model.cost_reward(rw), gamma=0.99)[0])
        if sp.k == 2:
            ref["frontier"] = model.frontier()
        out[name] = ref
        print(name, f"{time.time() - t0:.0f}s", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
