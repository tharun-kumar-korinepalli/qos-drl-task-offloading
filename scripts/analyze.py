import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from qos_offloading import ExactModel, Rules
from qos_offloading import study as S
from qos_offloading.rules import tune_reservation
from qos_offloading.stats import describe, holm, mann_whitney, sign_test_vs_constant, vargha_delaney_a12
from qos_offloading.training import TABLES, load

SUMMARY = ROOT / "results" / "summary"
REF = json.loads((SUMMARY / "references.json").read_text())
PAIRS = [("weighted", "flat"), ("corrected", "flat"), ("corrected", "weighted")]


def metrics(recs, src="exact"):
    k = len(recs[0][src]["miss"])
    out = {"cost": describe([r[src]["cost"] for r in recs])}
    for c in range(k):
        out[f"miss{c}"] = describe([r[src]["miss"][c] for r in recs])
    out["overall"] = describe([r[src]["overall"] for r in recs])
    if "latency" in recs[0][src]:
        for c in range(k):
            out[f"lat{c}"] = describe([r[src]["latency"][c] for r in recs])
    out["edge"] = describe([r[src]["edge"] for r in recs])
    out["costs"] = [r[src]["cost"] for r in recs]
    return out


def compare(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return {"diff": float(a.mean() - b.mean()), "a12": vargha_delaney_a12(a, b), "p": mann_whitney(list(a), list(b)),
            "lower": int((a < b).sum()), "n": len(a)}


def vs_rule(costs, c):
    b, w, p = sign_test_vs_constant(costs, c)
    return {"better": b, "worse": w, "p": p}


def group(jobs, prefix):
    out = {}
    for k, j in jobs.items():
        out.setdefault(k[:prefix], []).append(load(j))
    return out


def curves(groups):
    out = {}
    for key, recs in groups.items():
        if recs[0].get("curve"):
            arr = np.array([[c["cost"] for c in r["curve"]] for r in recs])
            out["|".join(map(str, key))] = {"timesteps": [c["timesteps"] for c in recs[0]["curve"]],
                                            "mean": arr.mean(0).tolist(),
                                            "lo": [describe(arr[:, i])["ci_low"] for i in range(arr.shape[1])],
                                            "hi": [describe(arr[:, i])["ci_high"] for i in range(arr.shape[1])]}
    return out


def with_holm(tests):
    for t, p in zip(tests, holm([t["p"] for t in tests])):
        t["p_holm"] = float(p)
    return tests


def nominal(best):
    g = group(S.main_jobs(best), 2)
    ref = REF["nominal"]
    rows = {}
    for key, rs in g.items():
        m = metrics(rs)
        m["gap_pct"] = 100 * (m["cost"]["mean"] / ref["optimum"]["cost"] - 1)
        m["vs_reservation"] = vs_rule(m["costs"], ref["reservation"]["cost"])
        m["vs_sla_rule"] = vs_rule(m["costs"], ref["sla_rule"]["cost"])
        m["mc_minus_exact"] = float(np.mean([r["mc"]["miss"][0] - r["exact"]["miss"][0] for r in rs]))
        rows["|".join(key)] = m
    tests = []
    for algo in ["ppo", "dqn", "a2c"]:
        for a, b in PAIRS:
            if (algo, a) in g and (algo, b) in g:
                for metric in ["cost", "miss0", "miss1"]:
                    get = (lambda r: r["exact"]["cost"]) if metric == "cost" else \
                        (lambda r, c=int(metric[-1]): r["exact"]["miss"][c])
                    tests.append({"algo": algo, "a": a, "b": b, "metric": metric,
                                  **compare([get(r) for r in g[(algo, a)]], [get(r) for r in g[(algo, b)]])})
    return {"rows": rows, "tests": with_holm(tests), "curves": curves(g)}


def systems(jobs):
    g = group(jobs, 2)
    rows = {}
    for key, rs in g.items():
        ref = REF[key[0]]
        m = metrics(rs)
        m["gap_pct"] = 100 * (m["cost"]["mean"] / ref["optimum"]["cost"] - 1)
        for rule in ["reservation", "node_sets", "sla_rule"]:
            m[f"vs_{rule}"] = vs_rule(m["costs"], ref[rule]["cost"])
        rows["|".join(key)] = m
    tests = [{"system": name, "a": a, "b": b, **compare([r["exact"]["cost"] for r in g[(name, a)]],
                                                       [r["exact"]["cost"] for r in g[(name, b)]])}
             for name in S.SYSTEMS for a, b in PAIRS]
    return {"rows": rows, "tests": with_holm(tests), "curves": curves(g)}


def published(best):
    g = group(S.published_jobs(best), 3)
    main = S.main_jobs(best)
    rows = {}
    for key, rs in g.items():
        src = "exact" if "exact" in rs[0] else "mc"
        m = metrics(rs, src)
        m["source"] = src
        m["gap_pct"] = 100 * (m["cost"]["mean"] / REF["nominal"]["optimum"]["cost"] - 1)
        m["vs_reservation"] = vs_rule(m["costs"], REF["nominal"]["reservation"]["cost"])
        if src == "mc":
            m["mc_se_mean"] = float(np.mean([r["mc"]["cost_se"] for r in rs]))
        m["final_penalties"] = np.mean([r["final_penalties"] for r in rs], axis=0).tolist()
        for rw in ["weighted", "flat"]:
            m[f"vs_ppo_{rw}"] = compare(m["costs"], [load(main[("ppo", rw, s)])["exact"]["cost"] for s in S.SEEDS])
        rows["|".join(map(str, key))] = m
    return {"rows": rows}


def shifts(best):
    main = S.main_jobs(best)
    tuned = S.shift_jobs(best, main)
    nom_rules = Rules(S.NOMINAL)
    th_nom, _ = tune_reservation(ExactModel(S.NOMINAL), nom_rules)
    rows = {}
    for shift, sp in S.SHIFTS.items():
        model = ExactModel(sp)
        row = {"optimum": model.evaluate(model.solve(model.cost_eval())[0]),
               "sla_rule_nominal_model": model.evaluate(model.policy_from_rule(nom_rules.sla)),
               "reservation_nominal": model.evaluate(model.policy_from_rule(nom_rules.reservation(th_nom)))}
        th, m = tune_reservation(model, Rules(sp))
        row["reservation_retuned"] = {**m, "thresholds": [None if not np.isfinite(x) else x for x in th]}
        for r in ["weighted", "corrected", "flat"]:
            evs = [model.evaluate(np.load(TABLES / f"{main[('ppo', r, s)].name()}.npz")["table"]) for s in S.SEEDS]
            costs = [e["cost"] for e in evs]
            row[f"ppo_{r}_nominal"] = {"cost": describe(costs), "miss0": describe([e["miss"][0] for e in evs]),
                                       "costs": costs}
        for r in ["weighted", "flat"]:
            row[f"ppo_{r}_finetuned"] = metrics([load(tuned[(shift, r, s)]) for s in S.SEEDS])
        rows[shift] = row
    return {"rows": rows}


def main():
    best = S.select_configs(load)
    best_sys = S.select_system_configs(load)
    out = {"tuning": {f"{a}|{r}": {**v, "config": [list(x) for x in v["config"]]} for (a, r), v in best.items()},
           "system_tuning": {f"{n}|{r}": {"index": v["index"], "score": v["score"], "all": v["all"]}
                             for (n, r), v in best_sys.items()},
           "nominal": nominal(best),
           "systems": systems(S.system_jobs(best_sys)),
           "systems_nominal_setting": systems(S.system_jobs_nominal_setting(best)),
           "published": published(best),
           "shifts": shifts(best)}
    (SUMMARY / "summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
