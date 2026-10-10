import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "results" / "summary"
S = json.loads((SUMMARY / "summary.json").read_text())
REF = json.loads((SUMMARY / "references.json").read_text())
RA = json.loads((SUMMARY / "reward_analysis.json").read_text())
SC = json.loads((SUMMARY / "dp_scaling.json").read_text())
NAMES = {"nominal": "Nominal", "hetero3": "Hetero3", "hetero3_3c": "Hetero3-3C", "large4_3c": "Large4-3C"}


def pc(x):
    return f"{100 * x:.1f}"


def ci(d, n=3):
    return f"{d['mean']:.{n}f} [{d['ci_low']:.{n}f}, {d['ci_high']:.{n}f}]"


def gap(c, o):
    return f"{100 * (c / o - 1):.1f}%"


def table(header, rows):
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows]


L = ["# Results", "",
     "Learned agents: mean over 10 seeds with 95% bootstrap interval. Rules and DP policies: exact. All values come "
     "from the exact model unless marked as simulated. SLA cost J = sum_c p_c e_c v_c.", "", "## Exact references", ""]
rows = []
for k, n in NAMES.items():
    r, d = REF[k], REF[k]["dp_rewards"]
    o = r["optimum"]["cost"]
    cell = lambda c: f"{c:.4f} ({gap(c, o)})"
    rows.append([n, f"{r['states']:,}", f"{o:.4f}", cell(r["sla_rule"]["cost"]), cell(r["reservation"]["cost"]),
                 cell(r["node_sets"]["cost"]), cell(d["weighted"]["cost"]), cell(d["corrected"]["cost"]),
                 cell(d["flat"]["cost"])])
L += table(["System", "States", "Optimum", "SLA rule", "Reservation", "Node sets", "DP weighted", "DP corrected",
            "DP flat"], rows)

L += ["", "## Operator weight ratio (nominal system)", ""]
L += table(["Ratio", "Optimum", "Corrected penalties", "Corrected", "Weighted", "Flat"],
           [[r, f"{v['J_opt']:.4f}", f"{v['wc_corr']:.2f}, {v['wa_corr']:.2f}", gap(v["J_corrected"], v["J_opt"]),
             gap(v["J_matched"], v["J_opt"]), gap(v["J_flat"], v["J_opt"])] for r, v in RA["ratios"].items()])

L += ["", "## Size and time of the exact DP", ""]
L += table(["Edge nodes", "States", "Transitions", "Solve (s)", "Evaluate (s)", "Peak memory (MB)"],
           [[str(r["edge_nodes"]), f"{r['states']:,}", f"{r['transition_entries']:,}", f"{r['solve_s']:.2f}",
             f"{r['evaluate_s']:.2f}", f"{r['peak_rss_mb']:.0f}"] for r in SC])

L += ["", "## Tuning", ""]
L += [f"- {k}: configuration {v['index']} {dict(v['config'])}, expected return {v['score']:.1f}"
      for k, v in S["tuning"].items()]
L += [f"- {k}: configuration {v['index']}, expected return {v['score']:.1f}" for k, v in S["system_tuning"].items()]

L += ["", "## Nominal system, 100,000 steps", ""]
L += table(["Agent", "v_crit (%)", "v_ana (%)", "SLA cost", "Above optimum", "Beats reservation", "Beats SLA rule"],
           [[k, pc(v["miss0"]["mean"]), pc(v["miss1"]["mean"]), ci(v["cost"]), f"{v['gap_pct']:.1f}%",
             f"{v['vs_reservation']['better']}/10 (p={v['vs_reservation']['p']:.3f})",
             f"{v['vs_sla_rule']['better']}/10"] for k, v in S["nominal"]["rows"].items()])
L += [""] + table(["Algo", "Rewards", "Metric", "Diff", "A12", "p (Holm)", "Lower"],
                  [[t["algo"], f"{t['a']} vs {t['b']}", t["metric"], f"{t['diff']:+.4f}", f"{t['a12']:.2f}",
                    f"{t['p_holm']:.4f}", f"{t['lower']}/{t['n']}"] for t in S["nominal"]["tests"]])

for key, title in [("systems", "Larger systems, PPO tuned per system"),
                   ("systems_nominal_setting", "Larger systems, PPO with the nominal setting")]:
    L += ["", f"## {title}", ""]
    L += table(["System, reward", "SLA cost", "Above optimum", "Beats reservation", "Beats node sets"],
               [[k, ci(v["cost"], 4), f"{v['gap_pct']:.1f}%", f"{v['vs_reservation']['better']}/10",
                 f"{v['vs_node_sets']['better']}/10"] for k, v in S[key]["rows"].items()])
    L += [""] + table(["System", "Rewards", "Diff", "A12", "p (Holm)", "Lower"],
                      [[t["system"], f"{t['a']} vs {t['b']}", f"{t['diff']:+.4f}", f"{t['a12']:.2f}",
                        f"{t['p_holm']:.4f}", f"{t['lower']}/{t['n']}"] for t in S[key]["tests"]])

L += ["", "## Tiered adaptive penalties (nominal system)", ""]
L += table(["Variant", "SLA cost", "Above optimum", "v_crit (%)", "Final penalties", "Scored by"],
           [[k, ci(v["cost"]), f"{v['gap_pct']:.1f}%", pc(v["miss0"]["mean"]),
             ", ".join(f"{x:.2f}" for x in v["final_penalties"]), "exact" if v["source"] == "exact" else "500 episodes"]
            for k, v in S["published"]["rows"].items()])

L += ["", "## Changes of the nominal system", ""]
rows = []
for s, r in S["shifts"]["rows"].items():
    o = r["optimum"]["cost"]
    rows.append([s, f"{o:.4f}"] + [gap(r[k]["cost"], o) for k in ["sla_rule_nominal_model", "reservation_nominal",
                                                                    "reservation_retuned"]]
                + [gap(r[k]["cost"]["mean"], o) for k in ["ppo_weighted_nominal", "ppo_weighted_finetuned",
                                                          "ppo_flat_nominal", "ppo_flat_finetuned"]])
L += table(["Change", "Optimum", "SLA rule", "Reservation, nominal", "Reservation, re-tuned", "PPO weighted, nominal",
            "PPO weighted, fine-tuned", "PPO flat, nominal", "PPO flat, fine-tuned"], rows)

(ROOT / "RESULTS.md").write_text("\n".join(L) + "\n")
