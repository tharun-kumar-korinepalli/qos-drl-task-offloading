"""Size, memory and time of the exact DP as the number of edge nodes grows (2 classes, random background)."""
import json
import resource
import time
from pathlib import Path

from qos_offloading.general import GeneralModel, SysParams

OUT = Path(__file__).resolve().parent.parent / "results" / "summary" / "dp_scaling.json"
rows = []
for n in [2, 3, 4, 5]:
    sp = SysParams(base_ms=(30.0,) * n, slope_ms=(15.0,) * n, drain=(0.6,) * n, background=0.6 * n - 1.0,
                   background_mode="bernoulli")
    t0 = time.time()
    m = GeneralModel(sp)
    tb = time.time() - t0
    nnz = sum(M.nnz for mats in m.P.values() for M in mats)
    t1 = time.time()
    pol, _ = m.solve(m.cost_eval())
    ts = time.time() - t1
    t2 = time.time()
    ev = m.evaluate(pol)
    te = time.time() - t2
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    rows.append({"edge_nodes": n, "states": m.S * m.K * m.T, "queue_states": m.S, "transition_entries": int(nnz),
                 "build_s": tb, "solve_s": ts, "evaluate_s": te, "peak_rss_mb": rss, "cost": ev["cost"]})
    print(rows[-1], flush=True)
    del m
OUT.write_text(json.dumps(rows, indent=1))
