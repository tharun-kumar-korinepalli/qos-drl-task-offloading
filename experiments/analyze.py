import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from qos_offloading import config as C
from qos_offloading import studies as S
from qos_offloading.dp import ExactModel, optimal_reward_policy, optimal_sla_policy, strip
from qos_offloading.reference import reference_policies
from qos_offloading.runner import ROOT, load
from qos_offloading.stats import (bootstrap_ci, describe, holm, mann_whitney, sign_test_vs_constant,
                                  vargha_delaney_a12)

OUT = ROOT / "results" / "summary"
TAB = ROOT / "results" / "tables"
METRICS = ["class_0_violation_rate", "class_1_violation_rate", "overall_violation_rate", "sla_cost_per_task",
           "class_0_avg_latency", "class_1_avg_latency", "edge_utilization"]
SHORT = {"class_0_violation_rate": "v_crit", "class_1_violation_rate": "v_ana",
         "overall_violation_rate": "v_all", "sla_cost_per_task": "cost",
         "class_0_avg_latency": "lat_crit", "class_1_avg_latency": "lat_ana", "edge_utilization": "edge"}
ALGO_NAME = {"ppo": "PPO", "dqn": "DQN", "a2c": "A2C"}


def collect(jobs, source="exact"):
    out, missing = {}, 0
    for k, j in jobs.items():
        r = load(j)
        if r is None:
            missing += 1
            continue
        out[k] = r if source == "record" else r[source]
    return out, missing


def per_group(records, group_fn):
    g = {}
    for k, r in records.items():
        g.setdefault(group_fn(k), []).append(r)
    return g


def vals(rs, m):
    return np.array([r[m] for r in rs], dtype=float)


def fmt_pct(d):
    return f"{100 * d['mean']:.1f} [{100 * d['ci_low']:.1f}, {100 * d['ci_high']:.1f}]"


def fmt_num(d, nd=2):
    return f"{d['mean']:.{nd}f} [{d['ci_low']:.{nd}f}, {d['ci_high']:.{nd}f}]"


def write_csv(path, rows, header):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main_study(ref):
    recs, miss = collect(S.main_jobs())
    groups = per_group(recs, lambda k: (k[0], k[1]))
    rows, summary = [], {}
    order = [("rr", "RR"), ("jsq", "JSQ"), ("threshold", "SLA rule"), ("reservation", "Reservation rule (tuned)")]
    for key, name in order:
        r = ref[key]
        rows.append([name, "1"] + [f"{100 * r[m]:.1f}" if "rate" in m or m == "edge_utilization"
                                   else f"{r[m]:.2f}" if m == "sla_cost_per_task" else f"{r[m]:.1f}" for m in METRICS])
        summary[name] = {m: r[m] for m in METRICS}
    for algo in S.ALGOS:
        for rw in ["flat", "weighted"]:
            rs = groups.get((algo, rw), [])
            if not rs:
                continue
            name = f"{ALGO_NAME[algo]} {rw}"
            d = {m: describe(vals(rs, m)) for m in METRICS}
            summary[name] = d
            rows.append([name, str(len(rs))] + [
                fmt_pct(d[m]) if "rate" in m or m == "edge_utilization"
                else fmt_num(d[m]) if m == "sla_cost_per_task" else fmt_num(d[m], 1) for m in METRICS])
    for key, name in [("dp_reward_flat", "Optimal for flat reward (DP)"),
                      ("dp_reward_weighted", "Optimal for weighted reward (DP)"),
                      ("dp_sla_optimal", "Optimal SLA cost (DP)")]:
        r = ref[key]
        rows.append([name, "exact"] + [f"{100 * r[m]:.1f}" if "rate" in m or m == "edge_utilization"
                                       else f"{r[m]:.3f}" if m == "sla_cost_per_task" else f"{r[m]:.1f}"
                                       for m in METRICS])
        summary[name] = {m: r[m] for m in METRICS}
    write_csv(TAB / "table_main.csv", rows, ["policy", "n"] + [SHORT[m] for m in METRICS])

    tests = []
    for algo in S.ALGOS:
        w, f = groups.get((algo, "weighted"), []), groups.get((algo, "flat"), [])
        if not w or not f:
            continue
        for m in ["class_0_violation_rate", "class_1_violation_rate", "sla_cost_per_task", "overall_violation_rate"]:
            a, b = vals(w, m), vals(f, m)
            tests.append({"algo": algo, "metric": m, "p": mann_whitney(a, b), "a12_w_gt_f": vargha_delaney_a12(a, b),
                          "no_overlap_w_below": bool(a.max() < b.min()),
                          "no_overlap_w_above": bool(a.min() > b.max()),
                          "seeds_w_lt_f": int((a < b).sum()) if len(a) == len(b) else None,
                          "n_w": len(a), "n_f": len(b),
                          "diff_mean": float(a.mean() - b.mean()),
                          "diff_ci": bootstrap_ci(a - b) if len(a) == len(b) else None})
    adj = holm([t["p"] for t in tests])
    for t, p in zip(tests, adj):
        t["p_holm"] = float(p)
    vs_rules = []
    for algo in S.ALGOS:
        w = groups.get((algo, "weighted"), [])
        if not w:
            continue
        for rule in ["threshold", "reservation"]:
            for m in ["class_0_violation_rate", "sla_cost_per_task"]:
                better, worse, p = sign_test_vs_constant(vals(w, m), ref[rule][m])
                vs_rules.append({"algo": algo, "rule": rule, "metric": m, "seeds_better": better,
                                 "seeds_worse": worse, "p_sign": p, "rule_value": ref[rule][m],
                                 "agent_mean": float(vals(w, m).mean())})
    gaps = {}
    opt = ref["dp_sla_optimal"]["sla_cost_per_task"]
    for name, d in summary.items():
        c = d["sla_cost_per_task"]["mean"] if isinstance(d["sla_cost_per_task"], dict) else d["sla_cost_per_task"]
        gaps[name] = {"cost": c, "gap_to_optimal_pct": 100 * (c / opt - 1)}
    return {"summary": summary, "tests": tests, "vs_rules": vs_rules, "optimality_gap": gaps, "missing": miss}


def penalty_study():
    recs, miss = collect(S.penalty_jobs())
    groups = per_group(recs, lambda k: (k[0], k[1]))
    model = ExactModel(S.BASE.eval_view())
    out = []
    for wc, wa in S.PENALTY_PAIRS:
        rs = groups.get((wc, wa), [])
        p = replace(S.BASE, critical_penalty=wc, analytics_penalty=wa)
        dp = strip(model.evaluate(optimal_reward_policy(model, p)))
        row = {"w_crit": wc, "w_ana": wa, "ratio": wc / wa, "n": len(rs), "dp_reward_optimal": dp}
        for m in METRICS:
            if rs:
                row[m] = describe(vals(rs, m))
        out.append(row)
    return {"rows": out, "missing": miss}


def load_study():
    recs, miss = collect(S.load_jobs())
    groups = per_group(recs, lambda k: (k[0], k[1]))
    out = []
    for d in S.DRAIN_PROBS:
        p = replace(S.BASE, drain_prob=d)
        ref = reference_policies(p)
        row = {"drain_prob": d, "utilization": p.utilization(),
               "reference": {k: v for k, v in ref.items() if k != "frontier"}}
        for rw in ["weighted", "flat"]:
            rs = groups.get((d, rw), [])
            row[rw] = {m: describe(vals(rs, m)) for m in METRICS} if rs else None
            row[rw + "_n"] = len(rs)
        w, f = groups.get((d, "weighted"), []), groups.get((d, "flat"), [])
        if w and f:
            row["p_crit"] = mann_whitney(vals(w, "class_0_violation_rate"), vals(f, "class_0_violation_rate"))
            row["p_cost"] = mann_whitney(vals(w, "sla_cost_per_task"), vals(f, "sla_cost_per_task"))
        out.append(row)
    return {"rows": out, "missing": miss}


def mix_study():
    recs, miss = collect(S.mix_jobs())
    groups = per_group(recs, lambda k: (k[0], k[1]))
    out = []
    for pc in S.CRITICAL_PROBS:
        p = replace(S.BASE, critical_prob=pc)
        ref = reference_policies(p)
        row = {"critical_prob": pc, "flat_penalty": p.matched_flat_penalty(),
               "reference": {k: v for k, v in ref.items() if k != "frontier"}}
        for rw in ["weighted", "flat"]:
            rs = groups.get((pc, rw), [])
            row[rw] = {m: describe(vals(rs, m)) for m in METRICS} if rs else None
            row[rw + "_n"] = len(rs)
        w, f = groups.get((pc, "weighted"), []), groups.get((pc, "flat"), [])
        if w and f:
            row["p_crit"] = mann_whitney(vals(w, "class_0_violation_rate"), vals(f, "class_0_violation_rate"))
            row["p_cost"] = mann_whitney(vals(w, "sla_cost_per_task"), vals(f, "sla_cost_per_task"))
        out.append(row)
    return {"rows": out, "missing": miss}


def lr_study():
    recs, miss = collect(S.lr_jobs())
    groups = per_group(recs, lambda k: (k[0], k[1]))
    out = []
    for lr in S.LEARNING_RATES:
        row = {"learning_rate": lr}
        for rw in ["weighted", "flat"]:
            rs = groups.get((lr, rw), [])
            row[rw] = {m: describe(vals(rs, m)) for m in METRICS} if rs else None
        out.append(row)
    return {"rows": out, "missing": miss}


def curve_study(ref):
    recs, miss = collect(S.curve_jobs(), source="record")
    groups = per_group(recs, lambda k: (k[0], k[1]))
    srecs, smiss = collect(S.dqn_scaled_jobs(), source="record")
    for (rw, s), r in srecs.items():
        groups.setdefault(("dqn-scaled", rw), []).append(r)
    miss += smiss
    out = {}
    for (algo, rw), rs in groups.items():
        steps = [c["timesteps"] for c in rs[0]["curve"]]
        arr = {m: np.array([[c[m] for c in r["curve"]] for r in rs]) for m in METRICS}
        out[f"{algo}_{rw}"] = {"timesteps": steps, "n": len(rs),
                               **{m: {"mean": arr[m].mean(0).tolist(),
                                      "ci": [bootstrap_ci(arr[m][:, i], n_boot=2000) for i in range(len(steps))]}
                                  for m in METRICS},
                               "final": {m: describe([r["exact"][m] for r in rs]) for m in METRICS},
                               "final_timesteps": int(min(r["timesteps_done"] for r in rs)),
                               "best": {m: describe(arr[m].min(1) if "rate" in m or m == "sla_cost_per_task"
                                                    else arr[m][:, -1]) for m in METRICS}}
    tests = []
    for algo in ["ppo", "dqn", "dqn-scaled"]:
        w, f = out.get(f"{algo}_weighted"), out.get(f"{algo}_flat")
        if not w or not f:
            continue
        wv = np.array([r["exact"]["sla_cost_per_task"] for r in groups[(algo, "weighted")]])
        fv = np.array([r["exact"]["sla_cost_per_task"] for r in groups[(algo, "flat")]])
        row = {"algo": algo, "p_cost_w_vs_f": mann_whitney(wv, fv), "w_mean": float(wv.mean()), "f_mean": float(fv.mean())}
        for rule in ["reservation", "threshold"]:
            b, wo, p = sign_test_vs_constant(wv, ref[rule]["sla_cost_per_task"])
            row[f"vs_{rule}"] = {"seeds_better": b, "seeds_worse": wo, "p_sign": p}
        row["gap_to_optimal_pct"] = float(100 * (wv.mean() / ref["dp_sla_optimal"]["sla_cost_per_task"] - 1))
        tests.append(row)
    return {"curves": out, "tests": tests, "missing": miss}


def mc_agreement():
    recs, _ = collect(S.main_jobs(), source="record")
    diffs = {m: [] for m in ["class_0_violation_rate", "class_1_violation_rate", "sla_cost_per_task"]}
    for r in recs.values():
        for m in diffs:
            diffs[m].append(r["mc"][m] - r["exact"][m])
    return {m: {"mean_diff": float(np.mean(v)), "max_abs_diff": float(np.max(np.abs(v))), "n": len(v)}
            for m, v in diffs.items() if v}


def mc_protocol_check():
    from qos_offloading.baselines import ReservationBaseline, SLAThresholdBaseline, TablePolicy
    from qos_offloading.envs import OffloadingEnv
    from qos_offloading.evaluation import evaluate_policy
    from qos_offloading.reference import sla_of
    ev = S.BASE.eval_view()
    model = ExactModel(ev)
    mk = lambda max_steps: OffloadingEnv(ev)
    rules = {"SLA rule": model.policy_from_obs_rule(SLAThresholdBaseline(sla_of(ev))),
             "Reservation rule": model.policy_from_obs_rule(ReservationBaseline(sla_of(ev), 0)),
             "Optimum (DP)": optimal_sla_policy(model)}
    out = {}
    for name, table in rules.items():
        ex = strip(model.evaluate(table))
        row = {"exact": ex["class_0_violation_rate"]}
        for label, n, s0 in [("mc_poster_20", C.EVAL_EPISODES, C.EVAL_SEED), ("mc_100", C.MC_EVAL_EPISODES, C.MC_EVAL_SEED),
                             ("mc_3000", 3000, 200_000)]:
            row[label] = evaluate_policy(TablePolicy(table), env_cls=mk, n_episodes=n, seed0=s0)["class_0_violation_rate"]
        out[name] = row
    return out


def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ref = reference_policies(S.BASE, frontier_points=201)
    res = {"reference": ref, "main": main_study(ref), "penalty": penalty_study(), "load": load_study(),
           "mix": mix_study(), "lr": lr_study(), "curves": curve_study(ref), "mc_vs_exact": mc_agreement(),
           "mc_protocol": mc_protocol_check()}
    (OUT / "summary.json").write_text(json.dumps(to_jsonable(res), indent=1))
    for k in ["main", "penalty", "load", "mix", "lr", "curves"]:
        print(f"{k}: missing runs = {res[k]['missing']}")
    print(f"written {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
