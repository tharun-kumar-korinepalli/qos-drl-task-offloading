import json

from qos_offloading.runner import ROOT

SUMMARY = ROOT / "results" / "summary" / "summary.json"
TAB = ROOT / "results" / "tables"
RESULTS_MD = ROOT / "RESULTS.md"
ROWS = ["RR", "JSQ", "SLA rule", "Reservation rule (tuned)", "A2C flat", "A2C weighted", "DQN flat", "DQN weighted",
        "PPO flat", "PPO weighted", "Optimal for flat reward (DP)", "Optimal for weighted reward (DP)",
        "Optimal SLA cost (DP)"]
COLS = [("class_0_violation_rate", "v_crit (%)", 100, 1), ("class_1_violation_rate", "v_ana (%)", 100, 1),
        ("overall_violation_rate", "All (%)", 100, 1), ("sla_cost_per_task", "SLA cost", 1, 3),
        ("class_0_avg_latency", "Lat. crit (ms)", 1, 1), ("class_1_avg_latency", "Lat. ana (ms)", 1, 1),
        ("edge_utilization", "Edge (%)", 100, 1)]
METRIC_NAME = {"class_0_violation_rate": "critical misses", "class_1_violation_rate": "analytics misses",
               "sla_cost_per_task": "SLA cost", "overall_violation_rate": "all misses"}


def cell(v, scale, nd, ci=True):
    if isinstance(v, dict):
        m = f"{scale * v['mean']:.{nd}f}"
        return f"{m} [{scale * v['ci_low']:.{nd}f}, {scale * v['ci_high']:.{nd}f}]" if ci else m
    return f"{scale * v:.{nd}f}"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def tex_table(header, rows, align):
    esc = lambda s: s.replace("%", "\\%").replace("_", "\\_")
    lines = ["\\begin{tabular}{@{}" + align + "@{}}", "\\toprule", " & ".join(esc(h) for h in header) + " \\\\",
             "\\midrule"]
    lines += [" & ".join(esc(c) for c in r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def pval(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def main():
    res = json.loads(SUMMARY.read_text())
    TAB.mkdir(parents=True, exist_ok=True)
    s = res["main"]["summary"]
    ref = res["reference"]
    md = ["# Results", "",
          "All numbers below are produced by `python run_all.py` and read from `results/summary/summary.json`. "
          "Learned agents are trained with 10 seeds for 20,000 steps and evaluated exactly on the full state space "
          "of the simulator, so evaluation adds no sampling noise. Values for learned agents are means with 95% "
          "bootstrap confidence intervals over seeds. Rules and DP policies are deterministic and evaluated exactly. "
          "SLA cost is the expected class-weighted miss penalty per task, 0.5 (10 v_crit + 2 v_ana).", ""]

    rows = []
    for name in ROWS:
        if name in s:
            rows.append([name] + [cell(s[name][k], sc, nd) for k, _, sc, nd in COLS])
    md += ["## Main comparison", "", md_table(["Policy"] + [c[1] for c in COLS], rows), ""]
    tex_rows = [[r[0]] + [cell(s[r[0]][k], sc, nd, ci=False) for k, _, sc, nd in COLS] for r in rows]
    (TAB / "table_main.tex").write_text(tex_table(["Policy"] + [c[1] for c in COLS], tex_rows, "l" + "r" * len(COLS)))
    md += [f"The tuned reservation rule sends analytics tasks to the edge only when the shorter queue holds at most "
           f"{ref['reservation']['analytics_queue_limit']} tasks.", ""]

    gap = res["main"]["optimality_gap"]
    rows = [[n, f"{gap[n]['cost']:.3f}", f"{gap[n]['gap_to_optimal_pct']:.0f}%"] for n in ROWS if n in gap]
    md += ["## Distance to the optimum", "", md_table(["Policy", "SLA cost", "Above optimum"], rows), ""]

    rows = []
    for t in res["main"]["tests"]:
        diff = f"{100 * t['diff_mean']:.1f}" if "rate" in t["metric"] else f"{t['diff_mean']:.3f}"
        rows.append([t["algo"].upper(), METRIC_NAME[t["metric"]], diff, f"{t['a12_w_gt_f']:.2f}", pval(t["p"]),
                     pval(t["p_holm"]), f"{t['seeds_w_lt_f']}/{t['n_w']}" if t["seeds_w_lt_f"] is not None else "-"])
    md += ["## Weighted against flat reward", "",
           "Difference is weighted minus flat (percentage points for rates). A12 is the Vargha-Delaney effect size, "
           "the probability that a weighted seed has a higher value than a flat seed. p values are two-sided "
           "Mann-Whitney U tests, exact when the samples have no ties and normal-approximated otherwise (only the "
           "A2C rows have ties), Holm-corrected over all rows. The last column counts seeds where the weighted agent "
           "is lower than the flat agent trained with the same seed.", "",
           md_table(["Algo", "Metric", "Diff", "A12", "p", "p (Holm)", "W lower"], rows), ""]
    (TAB / "table_tests.tex").write_text(tex_table(["Algo", "Metric", "Diff", "A12", "p", "p Holm", "W lower"], rows,
                                                   "llrrrrr"))

    rows = []
    for t in res["main"]["vs_rules"]:
        nd = 1 if "rate" in t["metric"] else 3
        sc = 100 if "rate" in t["metric"] else 1
        rows.append([t["algo"].upper(), t["rule"], METRIC_NAME[t["metric"]], f"{sc * t['agent_mean']:.{nd}f}",
                     f"{sc * t['rule_value']:.{nd}f}", f"{t['seeds_better']}/{t['seeds_better'] + t['seeds_worse']}",
                     pval(t["p_sign"])])
    md += ["## Weighted agents against rules", "",
           "Seeds better counts seeds where the agent has a lower value than the rule. p is a two-sided sign test.", "",
           md_table(["Algo", "Rule", "Metric", "Agent mean", "Rule", "Seeds better", "p"], rows), ""]

    rows = []
    for r in res["penalty"]["rows"]:
        if "class_0_violation_rate" not in r:
            continue
        rows.append([f"{r['w_crit']:g}:{r['w_ana']:g}", cell(r["class_0_violation_rate"], 100, 1),
                     cell(r["class_1_violation_rate"], 100, 1), cell(r["sla_cost_per_task"], 1, 3),
                     f"{100 * r['dp_reward_optimal']['class_0_violation_rate']:.1f}",
                     f"{100 * r['dp_reward_optimal']['class_1_violation_rate']:.1f}",
                     f"{r['dp_reward_optimal']['sla_cost_per_task']:.3f}"])
    md += ["## Penalty ratio (PPO, mean penalty 6)", "",
           md_table(["w_crit:w_ana", "v_crit (%)", "v_ana (%)", "SLA cost", "DP v_crit (%)", "DP v_ana (%)",
                     "DP SLA cost"], rows), ""]

    rows = []
    for r in res["load"]["rows"]:
        w, f, rf = r["weighted"], r["flat"], r["reference"]
        if not w or not f:
            continue
        rows.append([f"{r['drain_prob']:.1f}", f"{r['utilization']:.2f}", cell(f["class_0_violation_rate"], 100, 1),
                     cell(w["class_0_violation_rate"], 100, 1), cell(f["sla_cost_per_task"], 1, 3),
                     cell(w["sla_cost_per_task"], 1, 3), pval(r["p_cost"]),
                     f"{rf['dp_reward_flat']['sla_cost_per_task']:.3f}",
                     f"{rf['dp_reward_weighted']['sla_cost_per_task']:.3f}",
                     f"{rf['dp_sla_optimal']['sla_cost_per_task']:.3f}",
                     f"{rf['reservation']['sla_cost_per_task']:.3f}"])
    md += ["## Edge load (PPO)", "",
           "Load rho is the offered edge load, arrival rate over total drain rate of both edge nodes.", "",
           md_table(["Drain", "rho", "Flat v_crit (%)", "Weighted v_crit (%)", "Flat cost", "Weighted cost",
                     "p cost", "DP flat cost", "DP weighted cost", "Optimum", "Reservation"], rows), ""]

    rows = []
    for r in res["mix"]["rows"]:
        w, f, rf = r["weighted"], r["flat"], r["reference"]
        if not w or not f:
            continue
        rows.append([f"{100 * r['critical_prob']:.0f}", f"{r['flat_penalty']:.1f}",
                     cell(f["class_0_violation_rate"], 100, 1), cell(w["class_0_violation_rate"], 100, 1),
                     cell(f["sla_cost_per_task"], 1, 3), cell(w["sla_cost_per_task"], 1, 3), pval(r["p_cost"]),
                     f"{rf['dp_reward_flat']['sla_cost_per_task']:.3f}",
                     f"{rf['dp_reward_weighted']['sla_cost_per_task']:.3f}"])
    md += ["## Class mix (PPO)", "", "The flat penalty is matched to the average weighted penalty at each mix.", "",
           md_table(["Critical share (%)", "Flat penalty", "Flat v_crit (%)", "Weighted v_crit (%)", "Flat cost",
                     "Weighted cost", "p cost", "DP flat cost", "DP weighted cost"], rows), ""]

    rows = []
    for r in res["lr"]["rows"]:
        if not r["weighted"] or not r["flat"]:
            continue
        rows.append([f"{r['learning_rate']:g}", cell(r["flat"]["class_0_violation_rate"], 100, 1),
                     cell(r["weighted"]["class_0_violation_rate"], 100, 1),
                     cell(r["flat"]["sla_cost_per_task"], 1, 3), cell(r["weighted"]["sla_cost_per_task"], 1, 3)])
    md += ["## PPO learning rate (5 seeds)", "",
           md_table(["Learning rate", "Flat v_crit (%)", "Weighted v_crit (%)", "Flat cost", "Weighted cost"], rows), ""]

    rows = []
    for k, c in sorted(res["curves"]["curves"].items()):
        f = c["final"]
        rows.append([k.replace("_", " "), str(c["n"]), cell(f["class_0_violation_rate"], 100, 1),
                     cell(f["class_1_violation_rate"], 100, 1), cell(f["sla_cost_per_task"], 1, 3)])
    if rows:
        md += ["## After 100,000 training steps (5 seeds)", "",
               "Values are for the final policy of each run. dqn-scaled is DQN with all rewards multiplied by 0.1, "
               "which leaves the optimal policy unchanged.", "",
               md_table(["Agent", "n", "v_crit (%)", "v_ana (%)", "SLA cost"], rows), ""]
        trows = []
        for t in res["curves"]["tests"]:
            trows.append([t["algo"], f"{t['w_mean']:.3f}", f"{t['f_mean']:.3f}", pval(t["p_cost_w_vs_f"]),
                          f"{t['vs_reservation']['seeds_better']}/5", f"{t['vs_threshold']['seeds_better']}/5",
                          f"{t['gap_to_optimal_pct']:.0f}%"])
        md += ["Weighted against flat SLA cost of the final policies (Mann-Whitney U, 5 against 5 seeds), seeds where "
               "the weighted agent beats each rule on SLA cost, and distance of the weighted agent to the optimum:", "",
               md_table(["Agent", "Weighted cost", "Flat cost", "p", "Beats reservation", "Beats SLA rule",
                         "Above optimum"], trows), ""]

    a = res["mc_vs_exact"]
    if a:
        md += ["## Check of the exact evaluation", "",
               "Every learned agent is also simulated on the same 100 held-out episodes. Mean and largest difference "
               "between simulation and exact evaluation over all main runs:", "",
               md_table(["Metric", "Mean difference", "Largest difference", "Runs"],
                        [[METRIC_NAME[m], f"{v['mean_diff']:.4f}", f"{v['max_abs_diff']:.4f}", str(v["n"])]
                         for m, v in a.items()]), "",
               "The differences come from the finite test set, not from the exact model. All agents share the same "
               "100 episodes, so they share the same offset. The same offset appears for fixed policies, and it "
               "disappears with 3,000 episodes. Critical miss rate (%) of three fixed policies:", "",
               md_table(["Policy", "Exact", "20 episodes (poster protocol)", "100 episodes", "3,000 episodes"],
                        [[n, f"{100 * r['exact']:.2f}", f"{100 * r['mc_poster_20']:.2f}", f"{100 * r['mc_100']:.2f}",
                          f"{100 * r['mc_3000']:.2f}"] for n, r in res["mc_protocol"].items()]), ""]
    RESULTS_MD.write_text("\n".join(md) + "\n")
    print(f"written {RESULTS_MD} and tables in {TAB}")


if __name__ == "__main__":
    main()
