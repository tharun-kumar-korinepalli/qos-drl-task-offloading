"""Writes RESULTS_v3.md from results/summary/summary_v3.json and the exact reference files."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
S = json.loads((ROOT / "results" / "summary" / "summary_v3.json").read_text())
RA = json.loads((ROOT / "results" / "summary" / "reward_analysis.json").read_text())
SC = json.loads((ROOT / "results" / "summary" / "dp_scaling.json").read_text())
REF = S["reference"]
SYS = {"nominal": "Nominal", "hetero3": "Hetero3", "hetero3_3c": "Hetero3-3C", "large4_3c": "Large4-3C"}


def pc(x):
    return f"{100 * x:.1f}"


def ci(d, n=3):
    return f"{d['mean']:.{n}f} [{d['ci_low']:.{n}f}, {d['ci_high']:.{n}f}]"


def gap(c, o):
    return f"{100 * (c / o - 1):.1f}%"


L = ["# Results of the extended study (v3)", "",
     "Produced by `experiments/run_v3.py`, `experiments/analyze_v3.py` and this script. Learned agents: mean over 10 "
     "seeds with 95% bootstrap interval. Rules and DP policies: exact. Every value comes from the exact model unless "
     "marked as simulated. SLA cost J = sum_c p_c e_c v_c.", ""]

L += ["## Exact references", "", "| System | States | Optimum | SLA rule | Reservation | Reservation, node sets | DP weighted | DP corrected | DP flat |",
      "|---|---|---|---|---|---|---|---|---|"]
for k, n in SYS.items():
    r = REF[k]
    o = r["optimum"]["cost"]
    d = r["dp_rewards"]
    L.append(f"| {n} | {r['states']:,} | {o:.4f} | {r['sla_rule']['cost']:.4f} ({gap(r['sla_rule']['cost'], o)}) | "
             f"{r['reservation']['cost']:.4f} ({gap(r['reservation']['cost'], o)}) | "
             f"{r['reservation_sets']['cost']:.4f} ({gap(r['reservation_sets']['cost'], o)}) | "
             f"{d['weighted']['cost']:.4f} ({gap(d['weighted']['cost'], o)}) | {d['corrected']['cost']:.4f} "
             f"({gap(d['corrected']['cost'], o)}) | {d['flat']['cost']:.4f} ({gap(d['flat']['cost'], o)}) |")

L += ["", "## Operator weight ratio (nominal system, exact)", "",
      "| Ratio | Optimum | Corrected penalties | Corrected | Weighted | Flat |", "|---|---|---|---|---|---|"]
for r, v in RA["ratios"].items():
    o = v["J_opt"]
    L.append(f"| {r} | {o:.4f} | {v['wc_corr']:.2f}, {v['wa_corr']:.2f} | {gap(v['J_corrected'], o)} | "
             f"{gap(v['J_matched'], o)} | {gap(v['J_flat'], o)} |")

L += ["", "## DP size and time", "", "| Edge nodes | States | Transitions | Solve (s) | Evaluate (s) | Peak memory (MB) |",
      "|---|---|---|---|---|---|"]
for r in SC:
    L.append(f"| {r['edge_nodes']} | {r['states']:,} | {r['transition_entries']:,} | {r['solve_s']:.2f} | "
             f"{r['evaluate_s']:.2f} | {r['peak_rss_mb']:.0f} |")

L += ["", "## Tuning (nominal system)", ""]
for k, v in S["tuning"].items():
    L.append(f"- {k}: configuration {v['index']} {dict(v['config'])}, mean expected return {v['score']:.1f}")
if "scenario_tuning" in S:
    L += ["", "Per-system PPO tuning (configurations 4, 0, 5):", ""]
    for k, v in S["scenario_tuning"].items():
        L.append(f"- {k}: configuration {v['index']}, mean expected return {v['score']:.1f}")

L += ["", "## Nominal system, 100,000 steps", "", "| Agent | v_crit (%) | v_ana (%) | SLA cost | Above optimum | Beats reservation | Beats SLA rule |",
      "|---|---|---|---|---|---|---|"]
for k, v in S["main"]["rows"].items():
    L.append(f"| {k} | {pc(v['miss0']['mean'])} | {pc(v['miss1']['mean'])} | {ci(v['cost'])} | {v['gap_pct']:.1f}% | "
             f"{v['vs_reservation']['better']}/10 (p={v['vs_reservation']['p']:.3f}) | {v['vs_sla_rule']['better']}/10 |")
L += ["", "| Algo | Rewards | Metric | Diff | A12 | p (Holm) | Lower |", "|---|---|---|---|---|---|---|"]
for t in S["main"]["tests"]:
    L.append(f"| {t['algo']} | {t['a']} vs {t['b']} | {t['metric']} | {t['diff']:+.4f} | {t['a12']:.2f} | "
             f"{t['p_holm']:.4f} | {t['lower']}/{t['n']} |")

for key, title in [("scenarios", "Larger systems, PPO tuned per system"),
                   ("scenarios_nominal_settings", "Larger systems, PPO with the nominal setting")]:
    if key not in S or not S[key]["rows"]:
        continue
    L += ["", f"## {title}", "", "| System, reward | SLA cost | Above optimum | Beats reservation | Beats node-set rule |",
          "|---|---|---|---|---|"]
    for k, v in S[key]["rows"].items():
        L.append(f"| {k} | {ci(v['cost'], 4)} | {v['gap_pct']:.1f}% | {v['vs_reservation']['better']}/10 | "
                 f"{v['vs_reservation_sets']['better']}/10 (p={v['vs_reservation_sets']['p']:.3f}) |")
    L += ["", "| System | Rewards | Diff | A12 | p (Holm) | Lower |", "|---|---|---|---|---|---|"]
    for t in S[key]["tests"]:
        L.append(f"| {t['scenario']} | {t['a']} vs {t['b']} | {t['diff']:+.4f} | {t['a12']:.2f} | {t['p_holm']:.4f} | "
                 f"{t['lower']}/{t['n']} |")

if S["published"]["rows"]:
    L += ["", "## Published reward designs (nominal system)", "",
          "| Variant | SLA cost | Above optimum | v_crit (%) | Final penalties | Scored by |", "|---|---|---|---|---|---|"]
    for k, v in S["published"]["rows"].items():
        L.append(f"| {k} | {ci(v['cost'])} | {v['gap_pct']:.1f}% | {pc(v['miss0']['mean'])} | "
                 f"{', '.join(f'{x:.2f}' for x in v['final_penalties'])} | {v['source']} |")

L += ["", "## Changes of the nominal system (exact)", "",
      "| Change | Optimum | SLA rule | Reservation, nominal | Reservation, re-tuned | PPO weighted, nominal | PPO weighted, fine-tuned | PPO flat, nominal | PPO flat, fine-tuned |",
      "|---|---|---|---|---|---|---|---|---|"]
for s, r in S["robustness"]["rows"].items():
    o = r["optimum"]["cost"]
    cells = [f"{o:.4f}"] + [gap(r[k]["cost"], o) for k in ["sla_rule_nominal_model", "reservation_nominal",
                                                            "reservation_retuned"]]
    for k in ["ppo_weighted_nominal", "ppo_weighted_finetuned", "ppo_flat_nominal", "ppo_flat_finetuned"]:
        cells.append(gap(r[k]["cost"]["mean"], o) if k in r else "n/a")
    L.append(f"| {s} | " + " | ".join(cells) + " |")

(ROOT / "RESULTS_v3.md").write_text("\n".join(L) + "\n")
print("wrote RESULTS_v3.md")
