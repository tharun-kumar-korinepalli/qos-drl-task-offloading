"""Analysis of the revision experiments: 2M-step PPO in Large4-3C and the PPO-Lagrangian baseline.

    python scripts/analyze_revision.py

Writes results/summary/revision.json and prints every number that the red placeholders of the paper need
(Sec. V-E, the abstract, the conclusion and the runtime sentence of Sec. IV).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from qos_offloading import study as S
from qos_offloading.stats import describe, holm, mann_whitney, sign_test_vs_constant, vargha_delaney_a12
from qos_offloading.training import load

SUMMARY = ROOT / "results" / "summary"
REF = json.loads((SUMMARY / "references.json").read_text())
TOL = 1e-3   # a seed "meets" a target if its exact miss rate is at most target + 0.1 percentage points


def cost_stats(recs):
    costs = [r["exact"]["cost"] for r in recs]
    k = len(recs[0]["exact"]["miss"])
    return {"cost": describe(costs), "costs": costs,
            "miss": [describe([r["exact"]["miss"][c] for r in recs]) for c in range(k)]}


def compare(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return {"diff": float(a.mean() - b.mean()), "a12": vargha_delaney_a12(a, b), "p": mann_whitney(list(a), list(b)),
            "lower": int((a < b).sum()), "n": len(a)}


def vs_rule(costs, c):
    b, w, p = sign_test_vs_constant(costs, c)
    return {"better": b, "worse": w, "p": p}


def plateau(ts, mean, rel=0.02):
    """First curve point after which the mean cost never rises more than rel above its final value."""
    final = mean[-1]
    for i in range(len(mean)):
        if np.all(np.asarray(mean[i:]) <= final * (1 + rel)):
            return int(ts[i])
    return int(ts[-1])


def long_run(best_sys):
    jobs, short = S.long_jobs(best_sys), S.system_jobs(best_sys)
    ref = REF["large4_3c"]
    opt = ref["optimum"]["cost"]
    out = {}
    for r in ["weighted", "corrected"]:
        recs = [load(jobs[("large4_3c", r, s)]) for s in S.SEEDS]
        if any(x is None for x in recs):
            return None
        short_recs = [load(short[("large4_3c", r, s)]) for s in S.SEEDS]
        m = cost_stats(recs)
        m["gap_pct"] = 100 * (m["cost"]["mean"] / opt - 1)
        m["gap_pct_200k"] = 100 * (np.mean([x["exact"]["cost"] for x in short_recs]) / opt - 1)
        m["vs_200k"] = compare(m["costs"], [x["exact"]["cost"] for x in short_recs])
        m["vs_node_sets"] = vs_rule(m["costs"], ref["node_sets"]["cost"])
        m["vs_reservation"] = vs_rule(m["costs"], ref["reservation"]["cost"])
        arr = np.array([[c["cost"] for c in x["curve"]] for x in recs])
        ts = [c["timesteps"] for c in recs[0]["curve"]]
        mean = arr.mean(0)
        m["curve"] = {"timesteps": ts, "mean": mean.tolist(),
                      "gap_pct": (100 * (mean / opt - 1)).tolist()}
        m["plateau_2pct_at"] = plateau(ts, mean)
        m["steps_done"] = recs[0]["timesteps_done"]
        out[r] = m
    tests = [out[r]["vs_200k"] for r in out]
    for t, p in zip(tests, holm([t["p"] for t in tests])):
        t["p_holm"] = float(p)
    return out


def lagrangian(best, best_sys):
    etas = S.select_lagrange_eta(best, best_sys, load)
    if etas is None:
        return None
    jobs = S.lagrange_jobs(best, best_sys, etas)
    main, sysj = S.main_jobs(best), S.system_jobs(best_sys)
    out = {"eta": etas}
    tests = []
    for name in S.LAGRANGE_SYSTEMS:
        recs = [load(jobs[(name, s)]) for s in S.SEEDS]
        if any(x is None for x in recs):
            return None
        ref = REF[name]
        opt = ref["optimum"]["cost"]
        targets = list(S.optimum_miss(name)[:-1])
        m = cost_stats(recs)
        m["gap_pct"] = 100 * (m["cost"]["mean"] / opt - 1)
        m["targets"] = targets
        meets = [all(x["exact"]["miss"][c] <= targets[c] + TOL for c in range(len(targets))) for x in recs]
        m["seeds_meeting_all_targets"] = int(sum(meets))
        m["seeds_meeting_each_target"] = [int(sum(x["exact"]["miss"][c] <= targets[c] + TOL for x in recs))
                                          for c in range(len(targets))]
        lam = np.mean([x["final_penalties"] for x in recs], axis=0)
        beta = 1.0
        m["final_multipliers_mean"] = lam.tolist()
        # act as (lambda_c + beta) by Proposition 1; ratio of each class to the least critical one
        m["effective_ratio_to_last"] = [float((l + beta) / (lam[-1] + beta)) for l in lam]
        weighted = [load(main[("ppo", "weighted", s)]) if name == "nominal" else
                    load(sysj[(name, "weighted", s)]) for s in S.SEEDS]
        m["vs_ppo_weighted"] = compare(m["costs"], [x["exact"]["cost"] for x in weighted])
        rule = "reservation" if name == "nominal" else "node_sets"
        m[f"vs_{rule}"] = vs_rule(m["costs"], ref[rule]["cost"])
        tests.append(m["vs_ppo_weighted"])
        out[name] = m
    for t, p in zip(tests, holm([t["p"] for t in tests])):
        t["p_holm"] = float(p)
    return out


def fmt_ci(d):
    return f"{d['mean']:.3f} [{d['ci_low']:.3f}, {d['ci_high']:.3f}]"


def main():
    best = S.select_configs(load)
    best_sys = S.select_system_configs(load)
    out = {"long": long_run(best_sys), "lagrangian": lagrangian(best, best_sys)}
    rt = SUMMARY / "runtimes.json"
    out["runtimes"] = json.loads(rt.read_text()) if rt.exists() else None
    SUMMARY.mkdir(parents=True, exist_ok=True)
    (SUMMARY / "revision.json").write_text(json.dumps(out, indent=1))

    print("== Numbers for the red placeholders in main.tex ==")
    L = out["long"]
    if L:
        for r in ["weighted", "corrected"]:
            v = L[r]
            print(f"[V-E long] {r}: J = {fmt_ci(v['cost'])}, {v['gap_pct']:.0f}% above optimum "
                  f"(200k: {v['gap_pct_200k']:.0f}%), vs 200k p_holm={v['vs_200k']['p_holm']:.3f} "
                  f"A12={v['vs_200k']['a12']:.2f}, plateau(2%) at {v['plateau_2pct_at']:,} steps, "
                  f"beats node-set rule {v['vs_node_sets']['better']}/10, reservation {v['vs_reservation']['better']}/10")
    else:
        print("[V-E long] not finished")
    G = out["lagrangian"]
    if G:
        for name in S.LAGRANGE_SYSTEMS:
            v = G[name]
            rule = "reservation" if name == "nominal" else "node_sets"
            print(f"[V-E lagrangian] {name}: eta={G['eta'][name]['eta']}, J = {fmt_ci(v['cost'])}, "
                  f"{v['gap_pct']:.0f}% above optimum, all targets met in {v['seeds_meeting_all_targets']}/10 seeds "
                  f"(per class {v['seeds_meeting_each_target']}), vs weighted PPO p_holm={v['vs_ppo_weighted']['p_holm']:.3f} "
                  f"A12={v['vs_ppo_weighted']['a12']:.2f}, final multipliers "
                  f"{', '.join(f'{x:.2f}' for x in v['final_multipliers_mean'])}, effective ratio to last "
                  f"{', '.join(f'{x:.1f}' for x in v['effective_ratio_to_last'])}:1, beats {rule} {v[f'vs_{rule}']['better']}/10")
    else:
        print("[V-E lagrangian] not finished")
    R = out["runtimes"]
    if R and "large4_3c" in R:
        x = R["large4_3c"]
        print(f"[IV runtime] CPU {R['cpu']}: Large4-3C DP solve {x['solve_s']:.1f} s (model build {x['build_s']:.1f} s), "
              f"exact evaluation {x['evaluate_s']:.1f} s, PPO {x['ppo_steps']:,} steps {x['ppo_train_s'] / 60:.1f} min")
    else:
        print("[IV runtime] run scripts/runtimes.py first")


if __name__ == "__main__":
    main()
