import glob
import json
import os

import numpy as np
from scipy.stats import mannwhitneyu

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "poster_v1")
METRICS = ["class_0_violation_rate", "class_1_violation_rate", "overall_violation_rate",
           "sla_cost_per_task", "class_0_avg_latency", "class_1_avg_latency", "edge_utilization"]


def load(label):
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, f"scarce_{label}_*.json")))
    files = [f for f in files if os.path.basename(f)[len(f"scarce_{label}_"):-5].isdigit()]
    return [json.load(open(f)) for f in files]


def main():
    labels = ["rr", "jsq", "threshold", "dqn", "dqn_tuned_flat", "dqn_tuned", "flat", "weighted",
              "sweep_6_6", "sweep_8_4", "sweep_10_2", "sweep_11_1"]
    for label in labels:
        rs = load(label)
        if not rs:
            continue
        print(f"=== {label} (n={len(rs)})")
        for m in METRICS:
            v = np.array([r[m] for r in rs])
            print(f"  {m:24s} mean {v.mean():.4f} std {v.std():.4f} min {v.min():.4f} max {v.max():.4f}")
    w, f = load("weighted"), load("flat")
    dw, df = load("dqn_tuned"), load("dqn_tuned_flat")
    for m in ["class_0_violation_rate", "class_1_violation_rate", "sla_cost_per_task"]:
        a, b = [r[m] for r in w], [r[m] for r in f]
        print(f"MWU PPO weighted vs flat {m}: p={mannwhitneyu(a, b, method='exact').pvalue:.4f}")
        if dw and df:
            a, b = [r[m] for r in dw], [r[m] for r in df]
            print(f"MWU DQN weighted vs flat {m}: p={mannwhitneyu(a, b, method='exact').pvalue:.4f}")


if __name__ == "__main__":
    main()
