import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "TeX Gyre Termes"
plt.rcParams["font.size"] = 7
plt.rcParams["axes.linewidth"] = 0.6

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(HERE, "..", "results", "poster_v1"))
CRIT, ANA = "#8b1a1a", "#4a7fb5"


def load(label):
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, f"scarce_{label}_*.json")))
    files = [f for f in files if os.path.basename(f)[len(f"scarce_{label}_"):-5].isdigit()]
    return [json.load(open(f)) for f in files]


def stats(label, key):
    v = np.array([r[key] for r in load(label)]) * 100
    return v.mean(), (v.std() if len(v) > 1 else 0.0)


fig, (a, b) = plt.subplots(1, 2, figsize=(3.45, 1.55), gridspec_kw={"width_ratios": [1.35, 1]})

policies = [("weighted", "PPO weighted"), ("flat", "PPO flat"), ("dqn_tuned", "DQN weighted"),
            ("dqn_tuned_flat", "DQN flat"), ("threshold", "SLA rule"), ("jsq", "JSQ"), ("rr", "RR")]
y = np.arange(len(policies))[::-1]
h = 0.38
for yi, (label, name) in zip(y, policies):
    cm, cs = stats(label, "class_0_violation_rate")
    am, as_ = stats(label, "class_1_violation_rate")
    a.barh(yi + h / 2, cm, h, color=CRIT, xerr=cs, error_kw={"lw": 0.6, "capsize": 1.5})
    a.barh(yi - h / 2, am, h, color=ANA, xerr=as_, error_kw={"lw": 0.6, "capsize": 1.5})
a.set_yticks(y)
a.set_yticklabels([n for _, n in policies])
a.set_xlabel("SLA violations (%)")
a.set_xlim(0, 56)
from matplotlib.patches import Patch
a.legend(handles=[Patch(color=CRIT, label="critical tasks"), Patch(color=ANA, label="analytics tasks")],
         frameon=False, fontsize=6, loc="upper right", handlelength=1.2, borderpad=0.2)
a.set_title("(a) Violations per policy", fontsize=7)
a.spines[["top", "right"]].set_visible(False)

pairs = [(6, 6), (8, 4), (10, 2), (11, 1)]
labels = ["1:1", "2:1", "5:1", "11:1"]
x = np.arange(len(pairs))
c = [stats(f"sweep_{wc}_{wa}", "class_0_violation_rate") for wc, wa in pairs]
an = [stats(f"sweep_{wc}_{wa}", "class_1_violation_rate") for wc, wa in pairs]
b.errorbar(x, [v[0] for v in c], yerr=[v[1] for v in c], color=CRIT, marker="o", ms=3, lw=1, capsize=1.5,
           elinewidth=0.6, label="critical")
b.errorbar(x, [v[0] for v in an], yerr=[v[1] for v in an], color=ANA, marker="s", ms=3, lw=1, capsize=1.5,
           elinewidth=0.6, label="analytics")
b.set_xticks(x)
b.set_xticklabels(labels)
b.set_xlabel("penalty ratio crit : ana")
b.set_ylabel("SLA violations (%)")
b.set_ylim(0, 30)
b.legend(frameon=False, fontsize=6, loc="upper center", handlelength=1.2, ncol=2, columnspacing=0.8, borderpad=0.1)
b.set_title("(b) PPO, penalty ratio", fontsize=7)
b.spines[["top", "right"]].set_visible(False)

fig.tight_layout(pad=0.3, w_pad=1.0)
fig.savefig(os.path.join(HERE, "results.pdf"), bbox_inches="tight", pad_inches=0.01)
