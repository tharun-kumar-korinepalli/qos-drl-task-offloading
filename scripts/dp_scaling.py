import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from qos_offloading import ExactModel, System

OUT = ROOT / "results" / "summary" / "dp_scaling.json"

# CHANGE 1: replaces "import resource" (resource does not exist on Windows,
# and on Mac ru_maxrss is in bytes, not kilobytes)
try:
    import resource

    def peak_mb():
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 1024 / 1024 if sys.platform == "darwin" else r / 1024
except ImportError:
    def peak_mb():
        return float("nan")


def main():
    rows = []
    for n in [2, 3, 4, 5]:
        sp = System(base_ms=(30.0,) * n, slope_ms=(15.0,) * n, drain=(0.6,) * n, background=0.6 * n - 1.0,
                    background_mode="bernoulli")
        t0 = time.time()
        m = ExactModel(sp)
        build = time.time() - t0
        t1 = time.time()
        pol, _ = m.solve(m.cost_eval())
        solve = time.time() - t1
        t2 = time.time()
        ev = m.evaluate(pol)
        rows.append({"edge_nodes": n, "states": m.S * m.K * m.T, "queue_states": m.S,
                     "transition_entries": int(sum(M.nnz for mats in m.P.values() for M in mats)),
                     "build_s": build, "solve_s": solve, "evaluate_s": time.time() - t2,
                     "peak_rss_mb": peak_mb(),  # CHANGE 2: was resource.getrusage(...).ru_maxrss / 1024
                     "cost": ev["cost"]})
        print(rows[-1], flush=True)
        del m
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()