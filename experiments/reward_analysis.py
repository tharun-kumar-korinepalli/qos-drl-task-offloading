"""Exact analysis of the reward design: effective penalties, discounting, operator weights."""
import json
from pathlib import Path

import numpy as np

from qos_offloading import config as C
from qos_offloading.dp import ExactModel, optimal_sla_policy, strip
from qos_offloading.params import EnvParams

OUT = Path(__file__).resolve().parent.parent / "results" / "summary" / "reward_analysis.json"




def J(m, r, p=0.5):
    # operator cost with weights (r, 1), normalised so r=5 matches 0.5*(10 v_c + 2 v_a)
    return 2.0 * (p * r * m["class_0_violation_rate"] + (1 - p) * m["class_1_violation_rate"])


def reward_policy(model, wc, wa, success=C.SUCCESS_REWARD, cloud=C.CLOUD_USAGE_PENALTY, gamma=1.0):
    return model.solve(model.step_cost(wc, wa, success_reward=success, cloud_cost=cloud), gamma=gamma)[0]


def main():
    model = ExactModel(EnvParams().eval_view())
    ev = lambda pol: strip(model.evaluate(pol))
    out = {}

    # 1. effective penalties: +1 success bonus adds 1 to each penalty
    rows = {}
    rows["weighted 10:2 (paper)"] = ev(reward_policy(model, 10, 2))
    rows["costs 11:3 + cloud 0.1, no bonus"] = ev(reward_policy(model, 11, 3, success=0.0))
    rows["weighted 10:2, no bonus, no cloud cost"] = ev(reward_policy(model, 10, 2, success=0.0, cloud=0.0))
    rows["weighted 9:1 (effective 10:2)"] = ev(reward_policy(model, 9, 1))
    rows["weighted 10:2, no cloud cost"] = ev(reward_policy(model, 10, 2, cloud=0.0))
    rows["flat 6:6 (paper)"] = ev(reward_policy(model, 6, 6))
    rows["optimum"] = ev(optimal_sla_policy(model))
    out["effective"] = rows

    # 2. discounting as used by the learners
    disc = {}
    for g in (1.0, 0.99, 0.95, 0.9):
        disc[str(g)] = {"weighted": ev(reward_policy(model, 10, 2, gamma=g)),
                        "flat": ev(reward_policy(model, 6, 6, gamma=g))}
    out["discount"] = disc

    # 3. operator weight ratio r: optimum vs matched weighted reward (mean 6) vs flat 6
    ratios = {}
    flat = ev(reward_policy(model, 6, 6))
    for r in (1, 2, 3, 5, 10, 20, 50):
        wa = 12.0 / (r + 1)
        wc = r * wa
        opt = ev(model.solve(model.step_cost(float(r), 1.0))[0])
        matched = ev(reward_policy(model, wc, wa))
        # bonus-corrected: (wc + 1) : (wa + 1) = r with mean penalty 6
        wa2 = (14.0 / (r + 1)) - 1.0
        wc2 = 12.0 - wa2
        corrected = ev(reward_policy(model, wc2, wa2))
        ratios[str(r)] = {"wc": wc, "wa": wa, "wc_corr": wc2, "wa_corr": wa2,
                          "J_opt": J(opt, r), "J_matched": J(matched, r), "J_flat": J(flat, r),
                          "J_corrected": J(corrected, r),
                          "opt": opt, "matched": matched, "corrected": corrected}
    out["ratios"] = ratios
    OUT.write_text(json.dumps(out, indent=1))
    for k, m in rows.items():
        print(f"{k:40s} crit {100*m['class_0_violation_rate']:5.2f} ana {100*m['class_1_violation_rate']:5.2f} J {m['sla_cost_per_task']:.4f} edge {m['edge_utilization']:.3f}")
    for g, d in disc.items():
        print("gamma", g, "W", round(d["weighted"]["sla_cost_per_task"], 4), "F", round(d["flat"]["sla_cost_per_task"], 4))
    for r, d in ratios.items():
        print("r", r, "w", round(d["wc"], 2), round(d["wa"], 2), "Jopt %.4f Jmatched %.4f Jflat %.4f" % (d["J_opt"], d["J_matched"], d["J_flat"]),
              "gap matched %.1f%% corrected(%.2f:%.2f) %.2f%% flat %.1f%%" % (100 * (d["J_matched"] / d["J_opt"] - 1), d["wc_corr"], d["wa_corr"], 100 * (d["J_corrected"] / d["J_opt"] - 1), 100 * (d["J_flat"] / d["J_opt"] - 1)))


if __name__ == "__main__":
    main()
