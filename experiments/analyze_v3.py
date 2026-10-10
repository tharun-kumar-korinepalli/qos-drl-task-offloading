"""Builds results/summary/summary_v3.json from the runs of the extended study."""
import json
from pathlib import Path

import numpy as np

from qos_offloading import gexp
from qos_offloading import studies_v3 as V
from qos_offloading.general import GeneralModel, Rules, tune_reservation
from qos_offloading.stats import describe, holm, mann_whitney, sign_test_vs_constant, vargha_delaney_a12

ROOT = Path(__file__).resolve().parent.parent
SUM = ROOT / "results" / "summary"
REF = json.loads((SUM / "scenario_reference.json").read_text())


def metrics(recs, src="exact"):
    out = {}
    K = len(recs[0][src]["miss"])
    out["cost"] = describe([r[src]["cost"] for r in recs])
    for k in range(K):
        out[f"miss{k}"] = describe([r[src]["miss"][k] for r in recs])
    out["overall"] = describe([r[src]["overall"] for r in recs])
    if "latency" in recs[0][src]:
        for k in range(K):
            out[f"lat{k}"] = describe([r[src]["latency"][k] for r in recs])
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


def load_all(jobs):
    recs = {k: gexp.load(j) for k, j in jobs.items()}
    missing = [k for k, v in recs.items() if v is None]
    return recs, missing


def group(recs, prefix):
    out = {}
    for k, v in recs.items():
        if v is None:
            continue
        out.setdefault(k[:prefix], []).append(v)
    return out


def tuning(best):
    out = {}
    for (a, r), v in best.items():
        out[f"{a}|{r}"] = {"index": v["index"], "config": [list(x) for x in v["config"]], "score": v["score"],
                           "all": v["all"]}
    return out


def curves(groups):
    out = {}
    for key, recs in groups.items():
        if not recs[0].get("curve"):
            continue
        ts = [c["timesteps"] for c in recs[0]["curve"]]
        arr = np.array([[c["cost"] for c in r["curve"]] for r in recs])
        out["|".join(map(str, key))] = {"timesteps": ts, "mean": arr.mean(0).tolist(),
                                        "lo": [describe(arr[:, i])["ci_low"] for i in range(arr.shape[1])],
                                        "hi": [describe(arr[:, i])["ci_high"] for i in range(arr.shape[1])]}
    return out


def main_study(best):
    jobs = V.main_jobs(best)
    recs, missing = load_all(jobs)
    g = group(recs, 2)
    ref = REF["nominal"]
    opt = ref["optimum"]["cost"]
    res = {"missing": len(missing), "rows": {}, "tests": [], "curves": curves(g)}
    for key, rs in g.items():
        m = metrics(rs)
        m["gap_pct"] = 100 * (m["cost"]["mean"] / opt - 1)
        m["vs_reservation"] = vs_rule(m["costs"], ref["reservation"]["cost"])
        m["vs_sla_rule"] = vs_rule(m["costs"], ref["sla_rule"]["cost"])
        m["mc_minus_exact"] = float(np.mean([r["mc"]["miss"][0] - r["exact"]["miss"][0] for r in rs]))
        m["train_seconds"] = float(np.mean([r["train_seconds"] for r in rs]))
        res["rows"]["|".join(key)] = m
    tests = []
    for algo in ["ppo", "dqn", "a2c"]:
        for a, b in [("weighted", "flat"), ("corrected", "flat"), ("corrected", "weighted")]:
            if (algo, a) in g and (algo, b) in g:
                for metric in ["cost", "miss0", "miss1"]:
                    if metric == "cost":
                        xa, xb = [r["exact"]["cost"] for r in g[(algo, a)]], [r["exact"]["cost"] for r in g[(algo, b)]]
                    else:
                        k = int(metric[-1])
                        xa = [r["exact"]["miss"][k] for r in g[(algo, a)]]
                        xb = [r["exact"]["miss"][k] for r in g[(algo, b)]]
                    tests.append({"algo": algo, "a": a, "b": b, "metric": metric, **compare(xa, xb)})
    adj = holm([t["p"] for t in tests])
    for t, p in zip(tests, adj):
        t["p_holm"] = float(p)
    res["tests"] = tests
    return res


def scenarios(jobs):
    recs, missing = load_all(jobs)
    g = group(recs, 2)
    res = {"missing": len(missing), "rows": {}, "tests": [], "curves": curves(g)}
    for key, rs in g.items():
        ref = REF[key[0]]
        m = metrics(rs)
        m["gap_pct"] = 100 * (m["cost"]["mean"] / ref["optimum"]["cost"] - 1)
        for rule in ["reservation", "reservation_sets", "sla_rule"]:
            m[f"vs_{rule}"] = vs_rule(m["costs"], ref[rule]["cost"])
        m["train_seconds"] = float(np.mean([r["train_seconds"] for r in rs]))
        res["rows"]["|".join(key)] = m
    tests = []
    for name in V.SCENARIOS:
        for a, b in [("weighted", "flat"), ("corrected", "flat"), ("corrected", "weighted")]:
            if (name, a) in g and (name, b) in g:
                tests.append({"scenario": name, "a": a, "b": b, **compare(
                    [r["exact"]["cost"] for r in g[(name, a)]], [r["exact"]["cost"] for r in g[(name, b)]])})
    adj = holm([t["p"] for t in tests])
    for t, p in zip(tests, adj):
        t["p_holm"] = float(p)
    res["tests"] = tests
    return res


def published(best):
    jobs = V.published_jobs(best)
    recs, missing = load_all(jobs)
    g = group(recs, 3)
    res = {"missing": len(missing), "rows": {}}
    for key, rs in g.items():
        src = "exact" if "exact" in rs[0] else "mc"
        m = metrics(rs, src)
        ref = REF[key[0]]
        m["source"] = src
        m["gap_pct"] = 100 * (m["cost"]["mean"] / ref["optimum"]["cost"] - 1)
        m["vs_reservation"] = vs_rule(m["costs"], ref["reservation"]["cost"])
        if src == "mc":
            m["mc_se_mean"] = float(np.mean([r["mc"]["cost_se"] for r in rs]))
        m["final_penalties"] = np.mean([r["final_penalties"] for r in rs], axis=0).tolist()
        if key[0] == "nominal":
            mj = V.main_jobs(best)
            for rw in ["weighted", "flat"]:
                other = [gexp.load(mj[("ppo", rw, s)])["exact"]["cost"] for s in V.SEEDS]
                m[f"vs_ppo_{rw}"] = compare(m["costs"], other)
        res["rows"]["|".join(map(str, key))] = m
    return res


def robustness(best):
    main = V.main_jobs(best)
    ft_jobs = V.finetune_jobs(best, main)
    ft, missing = load_all(ft_jobs)
    nominal = V.NOMINAL
    nom_model = GeneralModel(nominal)
    nom_rules = Rules(nominal)
    th_nom, _ = tune_reservation(nom_model, nom_rules)
    res = {"missing": len(missing), "rows": {}}
    for shift, sp in V.SHIFTS.items():
        model = GeneralModel(sp)
        row = {}
        opt = model.evaluate(model.solve(model.cost_eval())[0])
        row["optimum"] = opt
        row["sla_rule_nominal_model"] = model.evaluate(model.policy_from_state_rule(nom_rules.sla))
        row["reservation_nominal"] = model.evaluate(model.policy_from_state_rule(nom_rules.reservation(th_nom)))
        true_rules = Rules(sp)
        row["sla_rule_true_model"] = model.evaluate(model.policy_from_state_rule(true_rules.sla))
        th, m = tune_reservation(model, true_rules)
        row["reservation_retuned"] = {**m, "thresholds": [None if not np.isfinite(x) else x for x in th]}
        for rname in ["weighted", "corrected", "flat"]:
            costs, miss0 = [], []
            for s in V.SEEDS:
                job = main[("ppo", rname, s)]
                p = gexp.GTABLES / f"{job.name()}.npz"
                if not p.exists():
                    continue
                tab = np.load(p)["table"]
                ev = model.evaluate(tab)
                costs.append(ev["cost"])
                miss0.append(ev["miss"][0])
            if costs:
                row[f"ppo_{rname}_nominal"] = {"cost": describe(costs), "miss0": describe(miss0), "costs": costs}
        for rname in ["weighted", "flat"]:
            rs = [ft[(shift, rname, s)] for s in V.SEEDS if ft.get((shift, rname, s))]
            if rs:
                row[f"ppo_{rname}_finetuned"] = metrics(rs)
        res["rows"][shift] = row
        print(shift, {k: round(v["cost"] if "cost" in v and not isinstance(v["cost"], dict) else v["cost"]["mean"], 4)
                      for k, v in row.items()}, flush=True)
    return res


def run():
    best = V.select_configs(gexp.load)
    out = {"tuning": tuning(best), "reference": REF}
    out["main"] = main_study(best)
    out["scenarios_nominal_settings"] = scenarios(V.scenario_jobs(best))
    bs = V.select_scen_configs(gexp.load)
    if bs is not None:
        out["scenario_tuning"] = {f"{n}|{r}": {"index": v["index"], "score": v["score"], "all": v["all"]}
                                  for (n, r), v in bs.items()}
        out["scenarios"] = scenarios(V.scenario_jobs_tuned(bs))
    else:
        out["scenarios"] = {"missing": -1, "rows": {}, "tests": [], "curves": {}}
    out["published"] = published(best)
    out["robustness"] = robustness(best)
    (SUM / "summary_v3.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    run()
