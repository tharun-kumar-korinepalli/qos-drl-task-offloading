"""Figures of the extended study, from results/summary/summary_v3.json and the run files."""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from qos_offloading import gexp
from qos_offloading import studies_v3 as V

HERE = Path(__file__).resolve().parent
SUM = HERE.parent / "results" / "summary"
OUT = Path(os.environ.get("FIG_DIR", HERE))

plt.rcParams.update({"font.family": ["TeX Gyre Termes", "Liberation Serif", "DejaVu Serif"], "font.size": 7,
                     "mathtext.fontset": "stix", "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                     "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42})
# validated categorical palette (fixed order: weighted, corrected, flat), plus line style as second encoding
COL = {"weighted": "#b03a2e", "corrected": "#b8860b", "flat": "#3a78b8"}
LS = {"weighted": "-", "corrected": "--", "flat": ":"}
NAME = {"weighted": "weighted", "corrected": "corrected", "flat": "flat"}
GRAY = "#555555"
COL1, COL2 = 3.45, 7.16


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)


def load():
    return json.loads((SUM / "summary_v3.json").read_text())


def fig_curves(S):
    cur = S["main"]["curves"]
    ref = S["reference"]["nominal"]
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 1.85), sharey=True)
    for ax, algo, title in zip(axes, ["ppo", "dqn", "a2c"], ["(a) PPO", "(b) DQN", "(c) A2C"]):
        for rw in ["weighted", "corrected", "flat"]:
            k = f"{algo}|{rw}"
            if k not in cur:
                continue
            c = cur[k]
            x = np.array(c["timesteps"]) / 1000
            ax.fill_between(x, c["lo"], c["hi"], color=COL[rw], alpha=0.15, lw=0)
            ax.plot(x, c["mean"], color=COL[rw], ls=LS[rw], lw=1.2)
        ax.axhline(ref["optimum"]["cost"], color="k", lw=0.8)
        ax.axhline(ref["reservation"]["cost"], color=GRAY, lw=0.8, ls=(0, (4, 2)))
        ax.axhline(ref["dp_rewards"]["flat"]["cost"], color=COL["flat"], lw=0.6, ls=(0, (1, 1.5)), alpha=0.7)
        ax.set_title(title, fontsize=7)
        ax.set_xlabel("training steps (thousands)")
        ax.set_ylim(0, 1.2)
        ax.set_xlim(0, 100)
        clean(ax)
    axes[0].set_ylabel("SLA cost $J$")
    h = [Line2D([], [], color=COL[r], ls=LS[r], lw=1.2, label=f"{NAME[r]} reward") for r in ["weighted", "corrected", "flat"]]
    h += [Line2D([], [], color="k", lw=0.8, label="optimum (DP)"),
          Line2D([], [], color=GRAY, lw=0.8, ls=(0, (4, 2)), label="tuned reservation rule"),
          Line2D([], [], color=COL["flat"], lw=0.6, ls=(0, (1, 1.5)), label="DP policy of flat reward")]
    fig.legend(handles=h, loc="lower center", ncol=6, fontsize=6, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "fig_curves_v3.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def fig_frontier(S):
    ref = S["reference"]["nominal"]
    best = V.select_configs(gexp.load)
    jobs = V.main_jobs(best)
    fr = np.array([[c, a] for _, c, a in ref["frontier"]]) * 100
    fr = fr[np.argsort(fr[:, 0])]
    fig, ax = plt.subplots(figsize=(COL1, 2.35))
    ax.fill_between(fr[:, 0], 0, fr[:, 1], color="#e6e6e6", lw=0)
    ax.plot(fr[:, 0], fr[:, 1], color="k", lw=0.9)
    mk = {"ppo": "o", "dqn": "s"}
    for algo in ["ppo", "dqn"]:
        for rw in ["weighted", "corrected", "flat"]:
            pts = []
            for s in V.SEEDS:
                j = jobs.get((algo, rw, s))
                r = gexp.load(j) if j else None
                if r:
                    pts.append((100 * r["exact"]["miss"][0], 100 * r["exact"]["miss"][1]))
            if pts:
                p = np.array(pts)
                ax.scatter(p[:, 0], p[:, 1], s=9, marker=mk[algo], facecolor=COL[rw] if algo == "ppo" else "none",
                           edgecolor=COL[rw], lw=0.7, zorder=3)
    for key, lab, m in [("sla_rule", "SLA rule", "D"), ("reservation", "reservation rule", "P")]:
        v = ref[key]
        ax.scatter(100 * v["miss"][0], 100 * v["miss"][1], marker=m, s=18, color=GRAY, zorder=4)
        ax.annotate(lab, (100 * v["miss"][0], 100 * v["miss"][1]), xytext=(4, 2), textcoords="offset points",
                    fontsize=6, color=GRAY)
    for key, lab in [("weighted", "DP weighted"), ("corrected", "DP corrected"), ("flat", "DP flat")]:
        v = ref["dp_rewards"][key]
        ax.scatter(100 * v["miss"][0], 100 * v["miss"][1], marker="*", s=40, color=COL[key], edgecolor="k", lw=0.4,
                   zorder=5)
    v = ref["optimum"]
    ax.scatter(100 * v["miss"][0], 100 * v["miss"][1], marker="*", s=40, color="k", zorder=6)
    ax.text(1.0, 2.0, "not reachable", fontsize=6, color=GRAY)
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 26)
    ax.set_xlabel("critical tasks missed (%)")
    ax.set_ylabel("analytics tasks missed (%)")
    clean(ax)
    h = [Line2D([], [], color="k", lw=0.9, label="optimal trade-off (DP)"),
         Line2D([], [], marker="*", color="k", ls="", ms=6, label="optimum"),
         Line2D([], [], marker="o", color=COL["weighted"], ls="", ms=3.5, label="PPO weighted"),
         Line2D([], [], marker="o", color=COL["corrected"], ls="", ms=3.5, label="PPO corrected"),
         Line2D([], [], marker="o", color=COL["flat"], ls="", ms=3.5, label="PPO flat"),
         Line2D([], [], marker="s", mfc="none", mec=GRAY, ls="", ms=3.5, label="DQN (hollow)"),
         Line2D([], [], marker="*", mfc=COL["weighted"], mec="k", ls="", ms=6, label="DP of each reward")]
    ax.legend(handles=h, fontsize=5.6, loc="upper right", handletextpad=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_frontier_v3.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


SYS_NAME = {"nominal": "Nominal", "hetero3": "Hetero3", "hetero3_3c": "Hetero3-3C", "large4_3c": "Large4-3C"}


def fig_scenarios(S):
    rows = S["scenarios"]["rows"]
    if not rows:
        return
    ref = S["reference"]
    systems = ["nominal", "hetero3", "hetero3_3c", "large4_3c"]
    items = [("SLA rule", "rule", "sla_rule"), ("reservation rule", "rule", "reservation"),
             ("node-set rule", "rule", "reservation_sets"),
             ("PPO flat", "agent", "flat"), ("PPO weighted", "agent", "weighted"), ("PPO corrected", "agent", "corrected")]
    fig, axes = plt.subplots(1, 4, figsize=(COL2, 1.75), sharey=True)
    for ax, sysname in zip(axes, systems):
        opt = ref[sysname]["optimum"]["cost"]
        y = np.arange(len(items))[::-1]
        for yi, (lab, kind, key) in zip(y, items):
            if kind == "rule":
                v = 100 * (ref[sysname][key]["cost"] / opt - 1)
                ax.plot([v], [yi], marker="D", color=GRAY, ms=3.5, ls="")
                continue
            k = f"{sysname}|{key}"
            src = S["main"]["rows"].get(f"ppo|{key}") if sysname == "nominal" else rows.get(k)
            if not src:
                continue
            m, lo, hi = (100 * (src["cost"][q] / opt - 1) for q in ("mean", "ci_low", "ci_high"))
            costs = 100 * (np.array(src["costs"]) / opt - 1)
            ax.scatter(costs, np.full(len(costs), yi) + np.linspace(-0.18, 0.18, len(costs)), s=3, color=COL[key],
                       alpha=0.6, lw=0)
            ax.plot([lo, hi], [yi, yi], color="k", lw=0.8)
            ax.plot([m], [yi], marker="|", color="k", ms=6)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_xscale("symlog", linthresh=10)
        ax.set_xlim(-2, 1000)
        ax.set_xticks([0, 10, 100, 1000])
        ax.set_xticklabels(["0", "10", "100", "1000"])
        ax.set_title(SYS_NAME[sysname], fontsize=7)
        ax.set_yticks(y)
        ax.set_yticklabels([i[0] for i in items])
        ax.set_xlabel("above optimum (%)")
        clean(ax)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(OUT / "fig_scenarios_v3.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def fig_shift(S):
    rows = S["robustness"]["rows"]
    shifts = list(rows)
    short = {s: s.split(" (")[0].replace("cloud meets analytics limit w.p. 0.5", "cloud meets analytics\nlimit w.p. 0.5")
             for s in shifts}
    items = [("reservation, nominal", lambda r: r["reservation_nominal"]["cost"], GRAY, "D", False),
             ("reservation, re-tuned", lambda r: r["reservation_retuned"]["cost"], GRAY, "D", True),
             ("PPO weighted, nominal", lambda r: r.get("ppo_weighted_nominal", {}).get("cost", {}).get("mean"),
              COL["weighted"], "o", False),
             ("PPO weighted, fine-tuned", lambda r: r.get("ppo_weighted_finetuned", {}).get("cost", {}).get("mean"),
              COL["weighted"], "o", True),
             ("PPO flat, nominal", lambda r: r.get("ppo_flat_nominal", {}).get("cost", {}).get("mean"),
              COL["flat"], "o", False),
             ("PPO flat, fine-tuned", lambda r: r.get("ppo_flat_finetuned", {}).get("cost", {}).get("mean"),
              COL["flat"], "o", True)]
    fig, ax = plt.subplots(figsize=(COL1, 2.2))
    y = np.arange(len(shifts))[::-1]
    for j, (lab, f, col, m, hollow) in enumerate(items):
        off = (j - (len(items) - 1) / 2) * 0.12
        for yi, s in zip(y, shifts):
            v = f(rows[s])
            if v is None:
                continue
            g = 100 * (v / rows[s]["optimum"]["cost"] - 1)
            ax.scatter([g], [yi + off], marker=m, s=12, facecolor="none" if hollow else col, edgecolor=col, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([short[s] for s in shifts])
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlim(-1, 400)
    ax.set_xticks([0, 10, 100])
    ax.set_xticklabels(["0", "10", "100"])
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("SLA cost above the optimum of the changed system (%)")
    clean(ax)
    h = [Line2D([], [], marker=m, ls="", mfc="none" if hollow else col, mec=col, ms=4, label=lab)
         for lab, _, col, m, hollow in items]
    ax.legend(handles=h, fontsize=5.6, loc="upper center", bbox_to_anchor=(0.42, -0.30), ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_shift_v3.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


if __name__ == "__main__":
    S = load()
    fig_curves(S)
    fig_frontier(S)
    fig_scenarios(S)
    fig_shift(S)
